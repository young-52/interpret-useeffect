/**
 * generate_reactivity_pilot.js
 *
 * Reactivity pilot 데이터셋 생성기.
 *
 * 설계 요약:
 *   - condition: useEffect 하나만 (5개 syntactic 환경으로 곱하지 않는다).
 *   - 한 변수 pair(= 기존 풀 항목의 dep/alt)당 2×2×2 = 8 cell.
 *       factor 1 state_role : dep가 reactive(useState) / alt가 reactive
 *       factor 2 decl_order : reactive 선언 먼저 / stable 선언 먼저
 *       factor 3 body_order : dep가 body 먼저 / alt가 body 먼저
 *   - 두 변수 모두 effect body에 등장한다 (body-presence confound 통제).
 *   - 모든 prefix는 prefix-only이며 '['로 끝난다. 측정 위치 = 마지막 '[' (baseline과 동일).
 *   - main metric은 Python 단계에서:  LD_reactive = logit(reactive_var) - logit(stable_var).
 *     generator는 logit을 계산하지 않고, 각 cell의 prefix와 target 식별자만 만든다.
 *   - target 식별자는 bare identifier로만 저장한다(§10.3). 실제 token id는 Python에서
 *     prefix 문맥으로 계산한다. baseline의 dep/alt naming은 끌어오지 않는다.
 *   - query key는 x/y로 고정한다(pool의 param은 이 pilot에서 쓰지 않는다).
 *
 * Usage:
 *   node generate_reactivity_pilot.js
 *   node generate_reactivity_pilot.js --input variable_names_validated.json --output reactivity_dataset.json
 */

const fs = require("fs");

// -------------------------------------------------------
// CLI
// -------------------------------------------------------
const args = process.argv.slice(2);
const getArg = (flag, def) => {
  const i = args.indexOf(flag);
  return i !== -1 ? args[i + 1] : def;
};
const INPUT = getArg("--input", "variable_names_validated.json");
const OUTPUT = getArg("--output", "reactivity_dataset.json");

const COMPONENT = "Component"; // §10.1 고정

// -------------------------------------------------------
// 헬퍼
// -------------------------------------------------------
function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

// -------------------------------------------------------
// factor 정의 (2×2×2)
// -------------------------------------------------------
const STATE_ROLES = ["dep_reactive", "alt_reactive"]; // dep/alt 중 누가 useState인가
const DECL_ORDERS = ["reactive_first", "stable_first"]; // 선언 줄 순서
const BODY_ORDERS = ["dep_first", "alt_first"]; // fetch body에서 dep/alt 등장 순서

// -------------------------------------------------------
// 한 cell(prefix) 생성
// -------------------------------------------------------
function buildCell(item, stateRole, declOrder, bodyOrder) {
  const { dep, alt, init } = item;

  // 어느 식별자가 reactive(useState)이고 어느 쪽이 stable(const)인가
  const reactiveVar = stateRole === "dep_reactive" ? dep : alt;
  const stableVar = stateRole === "dep_reactive" ? alt : dep;

  // 선언 줄 (stable은 component 내부 const, §0 고정 전제)
  const setter = "set" + capitalize(reactiveVar);
  const reactiveDecl = `  const [${reactiveVar}, ${setter}] = useState(${init});`;
  const stableDecl = `  const ${stableVar} = ${init};`;
  const decls =
    declOrder === "reactive_first"
      ? [reactiveDecl, stableDecl]
      : [stableDecl, reactiveDecl];

  // body 등장 순서는 식별자(dep/alt) 기준
  const bodyFirst = bodyOrder === "dep_first" ? dep : alt;
  const bodySecond = bodyOrder === "dep_first" ? alt : dep;
  const bodyLine = `    fetch(\`/api?x=\${${bodyFirst}}&y=\${${bodySecond}}\`);`;

  const prefix = [
    `function ${COMPONENT}() {`,
    ...decls,
    ``,
    `  useEffect(() => {`,
    bodyLine,
    `  }, [`, // ← prefix는 여기서 끝난다. measurement_pos = '['
  ].join("\n");

  // reactive 변수가 body에서 first인지 second인지 (order-dominated 해석용, §10.2)
  const reactiveBodyPos = bodyFirst === reactiveVar ? "first" : "second";

  return { prefix, reactiveVar, stableVar, reactiveBodyPos };
}

// -------------------------------------------------------
// invariant 검증
// -------------------------------------------------------
function assertCell(pairId, cellId, item, cell, stateRole) {
  const { dep, alt } = item;
  const { prefix, reactiveVar, stableVar } = cell;

  if (!prefix.endsWith("[")) {
    throw new Error(`[${cellId}] prefix가 '['로 끝나지 않습니다(§11.3).`);
  }
  // 두 변수 모두 effect body의 fetch에 등장해야 한다
  if (!prefix.includes(`\${${dep}}`) || !prefix.includes(`\${${alt}}`)) {
    throw new Error(`[${cellId}] dep/alt 두 변수가 body에 모두 등장하지 않습니다.`);
  }
  if (reactiveVar === stableVar) {
    throw new Error(`[${cellId}] reactive_var와 stable_var가 동일합니다.`);
  }
  // role label과 실제 useState 선언이 일치하는지
  const setter = "set" + capitalize(reactiveVar);
  if (!prefix.includes(`const [${reactiveVar}, ${setter}] = useState(`)) {
    throw new Error(`[${cellId}] state_role(${stateRole}) label과 실제 선언이 불일치합니다.`);
  }
  if (!prefix.includes(`const ${stableVar} = `)) {
    throw new Error(`[${cellId}] stable_var 선언이 없습니다.`);
  }
}

// -------------------------------------------------------
// 메인
// -------------------------------------------------------
const raw = fs.readFileSync(INPUT, "utf-8");
const items = JSON.parse(raw);

const rows = [];
items.forEach((item, idx) => {
  const pairId = `pair_${String(idx + 1).padStart(2, "0")}`;
  const { dep, alt, init } = item;

  let cellCounter = 0;
  for (const stateRole of STATE_ROLES) {
    for (const declOrder of DECL_ORDERS) {
      for (const bodyOrder of BODY_ORDERS) {
        cellCounter += 1;
        const cellId = `${pairId}_c${cellCounter}`;
        const cell = buildCell(item, stateRole, declOrder, bodyOrder);
        assertCell(pairId, cellId, item, cell, stateRole);

        rows.push({
          id: cellId,
          pair_id: pairId,
          type: "reactivity_pilot",
          condition: "useEffect",
          vars: { Component: COMPONENT, dep, alt, init }, // 식별자 고정 배정 A=dep, B=alt
          state_role: stateRole,
          decl_order: declOrder,
          body_order: bodyOrder,
          reactive_var: cell.reactiveVar, // useState 형태로 선언된 식별자
          stable_var: cell.stableVar, // component 내부 const 형태 식별자
          reactive_body_pos: cell.reactiveBodyPos, // first | second
          close_target: "]", // auxiliary diagnostic용
          prefix: cell.prefix,
        });
      }
    }
  }
});

// -------------------------------------------------------
// 전역 balance 검증 (§11.4): pair마다 8 cell, 8개가 완전한 2×2×2
// -------------------------------------------------------
const byPair = {};
for (const r of rows) (byPair[r.pair_id] ??= []).push(r);

const expectedCombos = new Set();
for (const s of STATE_ROLES)
  for (const d of DECL_ORDERS)
    for (const b of BODY_ORDERS) expectedCombos.add(`${s}|${d}|${b}`);

for (const [pairId, cells] of Object.entries(byPair)) {
  if (cells.length !== 8) {
    throw new Error(`[${pairId}] cell 수가 8이 아닙니다: ${cells.length}.`);
  }
  const got = new Set(cells.map((c) => `${c.state_role}|${c.decl_order}|${c.body_order}`));
  if (got.size !== 8) {
    throw new Error(`[${pairId}] 2×2×2 조합이 완전하지 않습니다(중복/누락).`);
  }
  for (const combo of expectedCombos) {
    if (!got.has(combo)) throw new Error(`[${pairId}] 누락 조합: ${combo}`);
  }
}

fs.writeFileSync(OUTPUT, JSON.stringify(rows, null, 2), "utf-8");
console.log(
  `✓ ${Object.keys(byPair).length} pair × 8 cell = ${rows.length} prefix 생성 → ${OUTPUT}`
);

// -------------------------------------------------------
// 샘플 출력 (첫 pair의 8 cell)
// -------------------------------------------------------
if (rows.length > 0) {
  const firstPair = rows[0].pair_id;
  console.log(`\n--- 샘플: ${firstPair}의 8 cell ---`);
  for (const r of rows.filter((r) => r.pair_id === firstPair)) {
    console.log(
      `\n[${r.id}] state_role=${r.state_role} decl_order=${r.decl_order} body_order=${r.body_order} | reactive=${r.reactive_var} stable=${r.stable_var} (reactive body ${r.reactive_body_pos})`
    );
    console.log(r.prefix);
  }
}
