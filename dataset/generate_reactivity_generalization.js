/**
 * generate_reactivity_generalization.js
 *
 * Generalization dataset generator for useState-form sensitivity in
 * dependency-array prediction.
 *
 * Design:
 *   - 45 variable pairs from the validated pilot pool.
 *   - 5 conditions:
 *       useEffect, useLayoutEffect, alias, alias_ctrl, subscribe.
 *   - Each (condition, pair) has the same 2x2x2 factorial as the pilot:
 *       state_role: dep_reactive / alt_reactive
 *       decl_order: reactive_first / stable_first
 *       body_order: dep_first / alt_first
 *   - Both candidate variables appear in the callback body.
 *   - Every prompt is prefix-only and ends exactly at the open bracket "[".
 *   - useEffect rows reuse the pilot template shape so their logits should
 *     reproduce the pilot when measured with the same models/tokenizers.
 *
 * Downstream sign convention lives in one helper in the logit/analysis script:
 *   D = logit(dep) - logit(alt)
 *   LD_stateform = (role_dep_useState_form ? +1 : -1) * D
 *
 * The legacy LD_reactive name is kept only as an output alias downstream.
 *
 * Usage:
 *   node generate_reactivity_generalization.js
 *   node generate_reactivity_generalization.js --input variable_names_validated.json --output reactivity_generalization_dataset.json
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
const OUTPUT = getArg("--output", "reactivity_generalization_dataset.json");

const COMPONENT = "Component";

// -------------------------------------------------------
// Helpers
// -------------------------------------------------------
function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function groupBy(rows, keyFn) {
  const groups = {};
  for (const row of rows) {
    const key = keyFn(row);
    (groups[key] ??= []).push(row);
  }
  return groups;
}

function countBy(rows, keyFn) {
  const counts = {};
  for (const row of rows) {
    const key = keyFn(row);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

// -------------------------------------------------------
// Conditions
// -------------------------------------------------------
const SUBSCRIBE_DEF = [
  "function subscribe(callback, values) {",
  "  callback();",
  "}",
].join("\n");

const CONDITION_ORDER = [
  "useEffect",
  "useLayoutEffect",
  "alias",
  "alias_ctrl",
  "subscribe",
];

const CONDITIONS = {
  useEffect: { preamble: "", callsite: "useEffect" },
  useLayoutEffect: { preamble: "", callsite: "useLayoutEffect" },
  alias: { preamble: "const myEffect = useEffect;\n\n", callsite: "myEffect" },
  alias_ctrl: {
    preamble: `${SUBSCRIBE_DEF}\n\nconst myEffect = subscribe;\n\n`,
    callsite: "myEffect",
  },
  subscribe: { preamble: `${SUBSCRIBE_DEF}\n\n`, callsite: "subscribe" },
};

// -------------------------------------------------------
// Factor definitions (2x2x2)
// -------------------------------------------------------
const STATE_ROLES = ["dep_reactive", "alt_reactive"];
const DECL_ORDERS = ["reactive_first", "stable_first"];
const BODY_ORDERS = ["dep_first", "alt_first"];

const EXPECTED_COMBOS = new Set();
for (const stateRole of STATE_ROLES)
  for (const declOrder of DECL_ORDERS)
    for (const bodyOrder of BODY_ORDERS)
      EXPECTED_COMBOS.add(`${stateRole}|${declOrder}|${bodyOrder}`);

// -------------------------------------------------------
// Cell generation
// -------------------------------------------------------
function buildCell(item, conditionName, stateRole, declOrder, bodyOrder) {
  const condition = CONDITIONS[conditionName];
  if (!condition) throw new Error(`Unknown condition: ${conditionName}`);

  const { dep, alt, init } = item;

  const useStateFormVar = stateRole === "dep_reactive" ? dep : alt;
  const constFormVar = stateRole === "dep_reactive" ? alt : dep;
  const roleDepUseStateForm = stateRole === "dep_reactive";

  const setter = "set" + capitalize(useStateFormVar);
  const useStateDecl = `  const [${useStateFormVar}, ${setter}] = useState(${init});`;
  const constDecl = `  const ${constFormVar} = ${init};`;
  const decls =
    declOrder === "reactive_first"
      ? [useStateDecl, constDecl]
      : [constDecl, useStateDecl];

  const bodyFirst = bodyOrder === "dep_first" ? dep : alt;
  const bodySecond = bodyOrder === "dep_first" ? alt : dep;
  const bodyLine = `    fetch(\`/api?x=\${${bodyFirst}}&y=\${${bodySecond}}\`);`;

  const componentPrefix = [
    `function ${COMPONENT}() {`,
    ...decls,
    ``,
    `  ${condition.callsite}(() => {`,
    bodyLine,
    `  }, [`,
  ].join("\n");

  const prefix = condition.preamble + componentPrefix;

  const useStateFormBodyPos = bodyFirst === useStateFormVar ? "first" : "second";
  const declDepFirst =
    (roleDepUseStateForm && declOrder === "reactive_first") ||
    (!roleDepUseStateForm && declOrder === "stable_first");

  return {
    prefix,
    bodyLine,
    depVar: dep,
    altVar: alt,
    useStateFormVar,
    constFormVar,
    reactiveVar: useStateFormVar,
    stableVar: constFormVar,
    roleDepUseStateForm,
    posDepFirst: bodyOrder === "dep_first",
    declDepFirst,
    useStateFormBodyPos,
    reactiveBodyPos: useStateFormBodyPos,
  };
}

// -------------------------------------------------------
// Invariants
// -------------------------------------------------------
function assertCell(cellId, conditionName, item, cell, stateRole) {
  const { dep, alt } = item;
  const {
    prefix,
    bodyLine,
    useStateFormVar,
    constFormVar,
    roleDepUseStateForm,
  } = cell;

  if (!prefix.endsWith("[")) {
    throw new Error(`[${cellId}] prefix must end exactly at "[".`);
  }
  if (prefix.endsWith("[ ") || prefix.endsWith("[\n")) {
    throw new Error(`[${cellId}] prefix has trailing whitespace after "[".`);
  }
  if (!bodyLine.includes(`\${${dep}}`) || !bodyLine.includes(`\${${alt}}`)) {
    throw new Error(`[${cellId}] both candidate variables must appear in the body.`);
  }
  if (useStateFormVar === constFormVar) {
    throw new Error(`[${cellId}] useState_form_var and const_form_var match.`);
  }
  if (roleDepUseStateForm !== (stateRole === "dep_reactive")) {
    throw new Error(`[${cellId}] role_dep_useState_form disagrees with state_role.`);
  }

  const setter = "set" + capitalize(useStateFormVar);
  if (!prefix.includes(`const [${useStateFormVar}, ${setter}] = useState(`)) {
    throw new Error(`[${cellId}] useState-form declaration is missing.`);
  }
  if (!prefix.includes(`const ${constFormVar} = `)) {
    throw new Error(`[${cellId}] const-form declaration is missing.`);
  }

  const condition = CONDITIONS[conditionName];
  if (!prefix.includes(`  ${condition.callsite}(() => {`)) {
    throw new Error(`[${cellId}] condition callsite is missing.`);
  }
}

function assertDataset(rows, pairCount) {
  const expectedTotal = pairCount * CONDITION_ORDER.length * EXPECTED_COMBOS.size;
  if (rows.length !== expectedTotal) {
    throw new Error(`Expected ${expectedTotal} rows, got ${rows.length}.`);
  }

  const byPairCondition = groupBy(
    rows,
    (row) => `${row.pair_id}|${row.condition}`
  );

  for (const [key, cells] of Object.entries(byPairCondition)) {
    if (cells.length !== 8) {
      throw new Error(`[${key}] expected 8 cells, got ${cells.length}.`);
    }

    const got = new Set(
      cells.map((row) => `${row.state_role}|${row.decl_order}|${row.body_order}`)
    );
    if (got.size !== 8) {
      throw new Error(`[${key}] 2x2x2 cells contain duplicates or omissions.`);
    }
    for (const combo of EXPECTED_COMBOS) {
      if (!got.has(combo)) throw new Error(`[${key}] missing combo: ${combo}`);
    }

    const bodyPos = countBy(cells, (row) => row.reactive_body_pos);
    if (bodyPos.first !== 4 || bodyPos.second !== 4) {
      throw new Error(
        `[${key}] reactive_body_pos is not 4/4 balanced: ` +
          JSON.stringify(bodyPos)
      );
    }

    const declBodyCombos = new Set(
      cells.map((row) => `${row.decl_order}|${row.body_order}`)
    );
    for (const declOrder of DECL_ORDERS) {
      for (const bodyOrder of BODY_ORDERS) {
        const combo = `${declOrder}|${bodyOrder}`;
        if (!declBodyCombos.has(combo)) {
          throw new Error(`[${key}] missing decl/body crossing: ${combo}`);
        }
      }
    }
  }

  const byPair = groupBy(rows, (row) => row.pair_id);
  for (const [pairId, cells] of Object.entries(byPair)) {
    const conditionSet = new Set(cells.map((row) => row.condition));
    for (const condition of CONDITION_ORDER) {
      if (!conditionSet.has(condition)) {
        throw new Error(`[${pairId}] missing condition: ${condition}`);
      }
    }
    const bodyPos = countBy(cells, (row) => row.reactive_body_pos);
    if (bodyPos.first !== 20 || bodyPos.second !== 20) {
      throw new Error(
        `[${pairId}] overall reactive_body_pos is not 20/20 balanced: ` +
          JSON.stringify(bodyPos)
      );
    }
  }
}

// -------------------------------------------------------
// Main
// -------------------------------------------------------
const raw = fs.readFileSync(INPUT, "utf-8");
const items = JSON.parse(raw);

if (items.length !== 45) {
  throw new Error(`Expected the same 45 validated pairs as the pilot, got ${items.length}.`);
}

const rows = [];
items.forEach((item, idx) => {
  const pairId = `pair_${String(idx + 1).padStart(2, "0")}`;
  const { dep, alt, init } = item;

  for (const conditionName of CONDITION_ORDER) {
    let cellCounter = 0;
    for (const stateRole of STATE_ROLES) {
      for (const declOrder of DECL_ORDERS) {
        for (const bodyOrder of BODY_ORDERS) {
          cellCounter += 1;
          const pilotCellId = `${pairId}_c${cellCounter}`;
          const cellId = `${pairId}_${conditionName}_c${cellCounter}`;
          const cell = buildCell(item, conditionName, stateRole, declOrder, bodyOrder);
          assertCell(cellId, conditionName, item, cell, stateRole);

          rows.push({
            id: cellId,
            pilot_cell_id: pilotCellId,
            pair_id: pairId,
            type: "reactivity_generalization",
            condition: conditionName,
            vars: { Component: COMPONENT, dep, alt, init },
            state_role: stateRole,
            decl_order: declOrder,
            body_order: bodyOrder,
            dep_var: cell.depVar,
            alt_var: cell.altVar,
            useState_form_var: cell.useStateFormVar,
            const_form_var: cell.constFormVar,
            reactive_var: cell.reactiveVar,
            stable_var: cell.stableVar,
            role_dep_useState_form: cell.roleDepUseStateForm,
            pos_dep_first: cell.posDepFirst,
            decl_dep_first: cell.declDepFirst,
            useState_form_body_pos: cell.useStateFormBodyPos,
            reactive_body_pos: cell.reactiveBodyPos,
            close_target: "]",
            prefix: cell.prefix,
          });
        }
      }
    }
  }
});

assertDataset(rows, items.length);

fs.writeFileSync(OUTPUT, JSON.stringify(rows, null, 2), "utf-8");
console.log(
  `OK: ${items.length} pair x 8 cell x ${CONDITION_ORDER.length} condition = ` +
    `${rows.length} prefixes -> ${OUTPUT}`
);

if (rows.length > 0) {
  const firstPair = rows[0].pair_id;
  console.log(`\nSample: ${firstPair}, useEffect condition`);
  for (const row of rows.filter(
    (r) => r.pair_id === firstPair && r.condition === "useEffect"
  )) {
    console.log(
      `\n[${row.id}] state_role=${row.state_role} ` +
        `decl_order=${row.decl_order} body_order=${row.body_order} ` +
        `useState_form=${row.useState_form_var} const_form=${row.const_form_var} ` +
        `(body ${row.useState_form_body_pos})`
    );
    console.log(row.prefix);
  }
}
