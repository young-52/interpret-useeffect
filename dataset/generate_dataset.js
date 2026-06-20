/**
 * generate_dataset.js
 *
 * variable_names_validated.json을 읽어 5개 조건(prefix-only) 템플릿으로 확장합니다.
 *
 * 연구 설계 근거: research_context_260620.md
 *   - §5  : 모든 입력은 prefix-only. dependency array 첫 항목 직전('[')에서 끊는다.
 *           measurement_pos = '[' 위치. target token은 입력에 포함되지 않는다.
 *   - §7  : 최종 기본 조건은 5개 (useEffect / useLayoutEffect / alias /
 *           alias_ctrl / subscribe).
 *   - §9  : 조건별 prefix-only 템플릿.
 *   - §10 : Component 이름 고정, target token id는 Python에서 prefix 문맥으로 계산.
 *   - §11 : baseline diagnostic은 logit(dep), logit(alt), logit(']') 세 개를 모두 사용.
 *
 * Usage:
 *   node generate_dataset.js
 *   node generate_dataset.js --input variable_names_validated.json --output dataset.json
 */

const fs = require("fs");

// -------------------------------------------------------
// CLI 인자
// -------------------------------------------------------
const args = process.argv.slice(2);
const getArg = (flag, def) => {
  const i = args.indexOf(flag);
  return i !== -1 ? args[i + 1] : def;
};
const INPUT = getArg("--input", "variable_names_validated.json");
const OUTPUT = getArg("--output", "dataset.json");

// -------------------------------------------------------
// 헬퍼
// -------------------------------------------------------

/** "dep" → "Dep" (setter 이름용: setDep) */
function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

// -------------------------------------------------------
// 조건 정의 (§7, §9)
// -------------------------------------------------------

// §9.4 / §9.5 공통: non-hook callback-array 함수 정의.
// subscribe 와 alias_ctrl 두 조건이 동일한 정의를 공유해야 한다.
const SUBSCRIBE_DEF = [
  "function subscribe(callback, values) {",
  "  callback();",
  "}",
].join("\n");

/**
 * 각 조건은 (1) component 앞에 붙는 preamble, (2) call-site 토큰만 다르다.
 * Component 본문 구조와 callback body는 모든 조건에서 동일하게 유지한다(§16 position 정렬).
 */
const CONDITIONS = {
  // §9.1 — 기준 조건
  useEffect: { preamble: "", callsite: "useEffect" },

  // §9.2 — React Effect Hook 계열 일반화 확인
  useLayoutEffect: { preamble: "", callsite: "useLayoutEffect" },

  // §9.3 — myEffect = useEffect (장거리 alias binding)
  alias: { preamble: "const myEffect = useEffect;\n\n", callsite: "myEffect" },

  // §9.4 — myEffect = subscribe (alias 구조/이름 통제)
  alias_ctrl: {
    preamble: `${SUBSCRIBE_DEF}\n\nconst myEffect = subscribe;\n\n`,
    callsite: "myEffect",
  },

  // §9.5 — non-React-Hook callback-array control
  subscribe: { preamble: `${SUBSCRIBE_DEF}\n\n`, callsite: "subscribe" },
};

// -------------------------------------------------------
// 템플릿 생성
// -------------------------------------------------------

/**
 * Component 본문을 prefix-only로 생성한다. 반드시 '[' 로 끝난다.
 * @param {string} bodyVar - callback body에 들어갈 변수 (clean=dep, corrupted=alt)
 */
function buildComponentBody({ Component, dep, alt, init, param, callsite, bodyVar }) {
  const Dep = capitalize(dep);
  return [
    `function ${Component}() {`,
    `  const [${dep}, set${Dep}] = useState(${init});`,
    `  const ${alt} = ${init};`,
    ``,
    `  ${callsite}(() => {`,
    `    fetch(\`/api?${param}=\${${bodyVar}}\`);`,
    `  }, [`, // ← prefix는 여기서 끝난다. measurement_pos = '[' (§5)
  ].join("\n");
}

/** 한 조건의 prefix(clean 또는 corrupted)를 만든다. */
function buildPrefix(conditionName, vars, bodyVar) {
  const spec = CONDITIONS[conditionName];
  const body = buildComponentBody({
    ...vars,
    callsite: spec.callsite,
    bodyVar,
  });
  return spec.preamble + body;
}

function makeDescription({ dep, alt, param }) {
  return (
    `prefix-only variable tracking — callback/effect body var: ` +
    `${dep} (clean) vs ${alt} (corrupted); param=${param}; ` +
    `prefix ends at open_bracket '['`
  );
}

// -------------------------------------------------------
// 검증 (생성 즉시 invariant 체크)
// -------------------------------------------------------

/**
 * prefix-only invariant:
 *   1. 모든 prefix는 '[' 로 끝나야 한다 (§5).
 *   2. 같은 조건 안에서 clean/corrupted는 줄 수가 같아야 한다 (§10.4의 사전 점검;
 *      토큰 길이 동일성은 Python Stage 1에서 토크나이저로 최종 확인).
 *   3. clean 과 corrupted 는 서로 달라야 한다.
 */
function assertPrefixInvariants(id, conditionName, clean, corrupted) {
  for (const [kind, text] of [["clean", clean], ["corrupted", corrupted]]) {
    if (!text.endsWith("[")) {
      throw new Error(
        `[${id}/${conditionName}/${kind}] prefix가 '['로 끝나지 않습니다. ` +
        `prefix-only 규약 위반(§5).`
      );
    }
  }
  if (clean === corrupted) {
    throw new Error(
      `[${id}/${conditionName}] clean과 corrupted가 동일합니다.`
    );
  }
  const cLines = clean.split("\n").length;
  const xLines = corrupted.split("\n").length;
  if (cLines !== xLines) {
    throw new Error(
      `[${id}/${conditionName}] clean/corrupted 줄 수 불일치: ${cLines} vs ${xLines}.`
    );
  }
}

// -------------------------------------------------------
// 메인
// -------------------------------------------------------

const raw = fs.readFileSync(INPUT, "utf-8");
const items = JSON.parse(raw);

const dataset = items.map((item, idx) => {
  const id = `sub_${String(idx + 1).padStart(2, "0")}`;

  // §10.1: Component 이름은 모든 샘플에서 고정. 입력값이 있어도 "Component"로 강제한다.
  const vars = {
    Component: "Component",
    dep: item.dep,
    alt: item.alt,
    init: item.init,
    param: item.param,
  };

  // 조건별 clean/corrupted prefix 생성 + invariant 검증
  const conditions = {};
  for (const conditionName of Object.keys(CONDITIONS)) {
    const clean = buildPrefix(conditionName, vars, vars.dep);
    const corrupted = buildPrefix(conditionName, vars, vars.alt);
    assertPrefixInvariants(id, conditionName, clean, corrupted);
    conditions[conditionName] = { clean, corrupted };
  }

  return {
    id,
    type: "substitution",
    description: makeDescription(vars),
    vars,

    // §10.3: bare identifier만 저장한다. 실제 target token id는 Python Stage 1에서
    // prefix 문맥(공백 포함 단일 토큰 여부 포함)으로 계산한다. " dep" 처럼 하드코딩하지 않는다.
    clean_target: vars.dep, // variable tracking metric의 clean target
    corrupted_target: vars.alt, // variable tracking metric의 corrupted target
    close_target: "]", // dep vs ] inclusion metric / baseline diagnostic용

    // 입력 파일에 토큰 검증 정보가 있으면 참고용으로 보존(있을 때만)
    token_forms: item._token_forms ?? null,

    conditions,
  };
});

fs.writeFileSync(OUTPUT, JSON.stringify(dataset, null, 2), "utf-8");
console.log(`✓ ${dataset.length}개 항목 × ${Object.keys(CONDITIONS).length}개 조건 생성 → ${OUTPUT}`);

// -------------------------------------------------------
// 샘플 출력 (첫 번째 항목, 전 조건의 clean prefix)
// -------------------------------------------------------
if (dataset.length > 0) {
  const s = dataset[0];
  console.log("\n--- 샘플 (첫 번째 항목) ---");
  console.log(`id: ${s.id}`);
  console.log(`description: ${s.description}`);
  console.log(`clean_target: ${s.clean_target}  corrupted_target: ${s.corrupted_target}  close_target: ${s.close_target}`);
  for (const name of Object.keys(CONDITIONS)) {
    console.log(`\n[${name} / clean]\n${s.conditions[name].clean}`);
  }
}
