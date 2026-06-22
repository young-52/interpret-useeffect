/**
 * generate_reactivity_ablation.js
 *
 * Ablation dataset generator for state-form preference in dependency-array /
 * callback-array completion.
 *
 * Holds declaration FIXED in every cell:
 *   target  = state-form-declared variable: `const [V, setV] = useState(init);`
 *   control = const-declared variable:      `const C = init;`
 * Both are always declared and in scope. What varies is which of the two the
 * callback body actually reads (Ablation C) or which array-introducing
 * wrapper follows a fixed body (Ablation B, optional).
 *
 * Ablation C (body-use, primary):
 *   axes per (pair, context): state_role(2) x decl_order(2) x bodyuse(4),
 *   where body_order(2) (dep_first/alt_first identity, same convention as the
 *   generalization generator) applies only to the bodyuse="both" cell.
 *   contexts: useEffect, subscribe.
 *
 * Ablation B (context-shape, optional):
 *   declaration AND body (`use(target, control);`, target always written
 *   first) held byte-identical across four wrapper contexts: useEffect,
 *   subscribe, plain_array, return_array. axes per pair:
 *   state_role(2) x decl_order(2) x context(4). No independent body_order
 *   factor (body text is fixed, see above).
 *
 * Sign convention lives downstream in the analysis script, same as the
 * generalization generator:
 *   D = logit(dep) - logit(alt)
 *   LD_stateform = (role_dep_useState_form ? +1 : -1) * D
 *
 * decl_block_end_offset (char offset into `prefix`) marks the end of the
 * declaration block so the analysis script can tokenize a decl-to-bracket
 * distance covariate without re-parsing the source text.
 *
 * Usage:
 *   node generate_reactivity_ablation.js
 *   node generate_reactivity_ablation.js --no-b
 *   node generate_reactivity_ablation.js --input variable_names_validated.json --output reactivity_ablation_dataset.json
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
const OUTPUT = getArg("--output", "reactivity_ablation_dataset.json");
const INCLUDE_B = !args.includes("--no-b");

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

function mentions(line, varName) {
  return new RegExp(`\\b${varName}\\b`).test(line);
}

// -------------------------------------------------------
// Contexts
// -------------------------------------------------------
const SUBSCRIBE_DEF = [
  "function subscribe(callback, values) {",
  "  callback();",
  "}",
].join("\n");

const C_CONTEXT_ORDER = ["useEffect", "subscribe"];
const B_CONTEXT_ORDER = ["useEffect", "subscribe", "plain_array", "return_array"];

const WRAPPERS = {
  useEffect: { preamble: "", kind: "callback", callsite: "useEffect" },
  subscribe: { preamble: `${SUBSCRIBE_DEF}\n\n`, kind: "callback", callsite: "subscribe" },
  plain_array: { preamble: "", kind: "statement", tail: "  const values = [" },
  return_array: { preamble: "", kind: "statement", tail: "  return [" },
};

// -------------------------------------------------------
// Factor definitions
// -------------------------------------------------------
const STATE_ROLES = ["dep_reactive", "alt_reactive"];
const DECL_ORDERS = ["reactive_first", "stable_first"];
const BODY_ORDERS = ["dep_first", "alt_first"];
const BODYUSE_ORDER = ["both", "target_only", "control_only", "neither"];

// -------------------------------------------------------
// Shared declaration block (identical across bodyuse cells and across
// contexts, by construction -- only depends on item/role/declOrder).
// -------------------------------------------------------
function buildDecls(item, stateRole, declOrder) {
  const { dep, alt, init } = item;
  const targetVar = stateRole === "dep_reactive" ? dep : alt;
  const controlVar = stateRole === "dep_reactive" ? alt : dep;
  const setter = "set" + capitalize(targetVar);
  const targetDecl = `  const [${targetVar}, ${setter}] = useState(${init});`;
  const controlDecl = `  const ${controlVar} = ${init};`;
  const decls =
    declOrder === "reactive_first" ? [targetDecl, controlDecl] : [controlDecl, targetDecl];
  return { decls, targetVar, controlVar, setter };
}

function buildBodyLine(callArgs, indent) {
  return `${indent}use(${callArgs.join(", ")});`;
}

function assemblePrefix(contextName, decls, bodyLine) {
  const wrapper = WRAPPERS[contextName];
  if (!wrapper) throw new Error(`Unknown context: ${contextName}`);

  const header = `function ${COMPONENT}() {`;
  const headerAndDecls = `${header}\n${decls.join("\n")}`;
  const declBlockEndOffset = wrapper.preamble.length + headerAndDecls.length;

  const lines = [header, ...decls, ``];
  if (wrapper.kind === "callback") {
    lines.push(`  ${wrapper.callsite}(() => {`, bodyLine, `  }, [`);
  } else {
    lines.push(bodyLine, wrapper.tail);
  }
  const componentPrefix = lines.join("\n");
  return { prefix: wrapper.preamble + componentPrefix, declBlockEndOffset };
}

function indentFor(contextName) {
  return WRAPPERS[contextName].kind === "callback" ? "    " : "  ";
}

// -------------------------------------------------------
// Common row assembly
// -------------------------------------------------------
function baseRow(item, pairId, ablation, contextName, stateRole, declOrder, decl, prefix, declBlockEndOffset) {
  const { dep, alt } = item;
  const { targetVar, controlVar } = decl;
  const roleDepUseStateForm = stateRole === "dep_reactive";
  const declDepFirst =
    (roleDepUseStateForm && declOrder === "reactive_first") ||
    (!roleDepUseStateForm && declOrder === "stable_first");

  return {
    pair_id: pairId,
    type: "reactivity_ablation",
    ablation,
    condition: contextName,
    vars: { Component: COMPONENT, dep, alt, init: item.init },
    state_role: stateRole,
    decl_order: declOrder,
    dep_var: dep,
    alt_var: alt,
    target_var: targetVar,
    control_var: controlVar,
    useState_form_var: targetVar,
    const_form_var: controlVar,
    role_dep_useState_form: roleDepUseStateForm,
    decl_dep_first: declDepFirst,
    decl_block_end_offset: declBlockEndOffset,
    close_target: "]",
    prefix,
  };
}

// -------------------------------------------------------
// Ablation C: body-use
// -------------------------------------------------------
function buildRowsC(item, pairId) {
  const rows = [];

  for (const contextName of C_CONTEXT_ORDER) {
    for (const stateRole of STATE_ROLES) {
      for (const declOrder of DECL_ORDERS) {
        const decl = buildDecls(item, stateRole, declOrder);
        const declText = decl.decls.join("\n");
        const indent = indentFor(contextName);

        for (const bodyuse of BODYUSE_ORDER) {
          const bodyOrders = bodyuse === "both" ? BODY_ORDERS : [null];

          for (const bodyOrder of bodyOrders) {
            let callArgs;
            if (bodyuse === "both") {
              const { dep, alt } = item;
              callArgs = bodyOrder === "dep_first" ? [dep, alt] : [alt, dep];
            } else if (bodyuse === "target_only") {
              callArgs = [decl.targetVar];
            } else if (bodyuse === "control_only") {
              callArgs = [decl.controlVar];
            } else {
              callArgs = [];
            }

            const bodyLine = buildBodyLine(callArgs, indent);
            const { prefix, declBlockEndOffset } = assemblePrefix(contextName, decl.decls, bodyLine);

            const cellId =
              `${pairId}_C_${contextName}_${stateRole}_${declOrder}_${bodyuse}` +
              (bodyOrder ? `_${bodyOrder}` : "");

            const row = baseRow(
              item,
              pairId,
              "C",
              contextName,
              stateRole,
              declOrder,
              decl,
              prefix,
              declBlockEndOffset
            );
            row.id = cellId;
            row.bodyuse = bodyuse;
            row.body_order = bodyOrder;
            row.pos_dep_first = bodyuse === "both" ? bodyOrder === "dep_first" : null;
            row.useState_form_body_pos =
              bodyuse === "both" ? (callArgs[0] === decl.targetVar ? "first" : "second") : null;
            row.reactive_var = decl.targetVar;
            row.stable_var = decl.controlVar;
            row.reactive_body_pos = row.useState_form_body_pos;

            assertCellC(cellId, item, decl, declText, bodyLine, bodyuse, row);
            rows.push(row);
          }
        }
      }
    }
  }

  return rows;
}

function assertCellC(cellId, item, decl, declText, bodyLine, bodyuse, row) {
  const { prefix } = row;
  const { dep, alt } = item;
  const { targetVar, controlVar } = decl;

  if (!prefix.endsWith("[")) throw new Error(`[${cellId}] prefix must end exactly at "[".`);
  if (prefix.endsWith("[ ") || prefix.endsWith("[\n")) {
    throw new Error(`[${cellId}] prefix has trailing whitespace after "[".`);
  }
  if (targetVar === controlVar) throw new Error(`[${cellId}] target_var and control_var match.`);

  const declBlock = prefix.slice(0, row.decl_block_end_offset);
  const setter = "set" + capitalize(targetVar);
  if (!declBlock.includes(`const [${targetVar}, ${setter}] = useState(`)) {
    throw new Error(`[${cellId}] target (state-form) declaration is missing or not in scope.`);
  }
  if (!declBlock.includes(`const ${controlVar} = `)) {
    throw new Error(`[${cellId}] control (const-form) declaration is missing or not in scope.`);
  }

  const hasTarget = mentions(bodyLine, targetVar);
  const hasControl = mentions(bodyLine, controlVar);
  if (bodyuse === "both") {
    if (!(mentions(bodyLine, dep) && mentions(bodyLine, alt))) {
      throw new Error(`[${cellId}] bodyuse=both must mention both dep and alt.`);
    }
  } else if (bodyuse === "target_only") {
    if (!hasTarget || hasControl) {
      throw new Error(`[${cellId}] bodyuse=target_only must mention target only.`);
    }
  } else if (bodyuse === "control_only") {
    if (!hasControl || hasTarget) {
      throw new Error(`[${cellId}] bodyuse=control_only must mention control only.`);
    }
  } else if (bodyuse === "neither") {
    if (hasTarget || hasControl) {
      throw new Error(`[${cellId}] bodyuse=neither must mention neither variable.`);
    }
  } else {
    throw new Error(`[${cellId}] unknown bodyuse: ${bodyuse}`);
  }
}

function assertDeclIdentityAcrossBodyuseC(rows) {
  const byKey = groupBy(
    rows,
    (r) => `${r.pair_id}|${r.condition}|${r.state_role}|${r.decl_order}`
  );
  for (const [key, cells] of Object.entries(byKey)) {
    const declBlocks = new Set(cells.map((r) => r.prefix.slice(0, r.decl_block_end_offset)));
    if (declBlocks.size !== 1) {
      throw new Error(`[${key}] declarations are not byte-identical across bodyuse cells.`);
    }
  }
}

function assertCrossingC(rows, pairCount) {
  const expectedTotal =
    pairCount *
    C_CONTEXT_ORDER.length *
    STATE_ROLES.length *
    DECL_ORDERS.length *
    (BODYUSE_ORDER.length - 1 + BODY_ORDERS.length); // 3 singleton bodyuse + 2 for "both"
  if (rows.length !== expectedTotal) {
    throw new Error(`Ablation C: expected ${expectedTotal} rows, got ${rows.length}.`);
  }

  const byPairContext = groupBy(rows, (r) => `${r.pair_id}|${r.condition}`);
  for (const [key, cells] of Object.entries(byPairContext)) {
    const expectedPerCell = STATE_ROLES.length * DECL_ORDERS.length;
    if (cells.length !== expectedPerCell * (BODYUSE_ORDER.length - 1 + BODY_ORDERS.length)) {
      throw new Error(`[${key}] unexpected ablation C cell count: ${cells.length}.`);
    }
    const byBodyuse = groupBy(cells, (r) => r.bodyuse);
    for (const bodyuse of BODYUSE_ORDER) {
      const group = byBodyuse[bodyuse] ?? [];
      const expected = bodyuse === "both" ? expectedPerCell * BODY_ORDERS.length : expectedPerCell;
      if (group.length !== expected) {
        throw new Error(`[${key}] bodyuse=${bodyuse} expected ${expected} cells, got ${group.length}.`);
      }
      const combos = new Set(
        group.map((r) => `${r.state_role}|${r.decl_order}|${r.body_order ?? ""}`)
      );
      if (combos.size !== expected) {
        throw new Error(`[${key}] bodyuse=${bodyuse} role/decl(/body_order) crossing has duplicates.`);
      }
    }
  }
}

// -------------------------------------------------------
// Ablation B (optional): context-shape
// -------------------------------------------------------
function buildRowsB(item, pairId) {
  const rows = [];

  for (const stateRole of STATE_ROLES) {
    for (const declOrder of DECL_ORDERS) {
      const decl = buildDecls(item, stateRole, declOrder);
      const declText = decl.decls.join("\n");
      // Body is fixed and identical across contexts: target always written
      // first, by construction (no independent body_order factor for B).
      const fixedArgs = [decl.targetVar, decl.controlVar];

      const variants = [];
      for (const contextName of B_CONTEXT_ORDER) {
        const indent = indentFor(contextName);
        const bodyLine = buildBodyLine(fixedArgs, indent);
        const { prefix, declBlockEndOffset } = assemblePrefix(contextName, decl.decls, bodyLine);

        const cellId = `${pairId}_B_${contextName}_${stateRole}_${declOrder}`;
        const row = baseRow(
          item,
          pairId,
          "B",
          contextName,
          stateRole,
          declOrder,
          decl,
          prefix,
          declBlockEndOffset
        );
        row.id = cellId;
        row.bodyuse = "both";
        row.body_order = stateRole === "dep_reactive" ? "dep_first" : "alt_first"; // derived, not a free factor
        row.pos_dep_first = stateRole === "dep_reactive";
        row.useState_form_body_pos = "first";
        row.reactive_var = decl.targetVar;
        row.stable_var = decl.controlVar;
        row.reactive_body_pos = "first";

        assertCellB(cellId, item, decl, bodyLine, row);
        variants.push({ row, declText, bodyLine });
        rows.push(row);
      }

      const declBlocks = new Set(
        variants.map((v) => {
          const preambleLen = WRAPPERS[v.row.condition].preamble.length;
          return v.row.prefix.slice(preambleLen, v.row.decl_block_end_offset);
        })
      );
      if (declBlocks.size !== 1) {
        throw new Error(
          `[${pairId}/${stateRole}/${declOrder}] Ablation B declarations are not identical across contexts.`
        );
      }
      // Indentation differs by nesting depth (callback wrapper vs. bare
      // statement); compare the call expression itself, not whitespace.
      const bodyLines = new Set(variants.map((v) => v.bodyLine.trim()));
      if (bodyLines.size !== 1) {
        throw new Error(
          `[${pairId}/${stateRole}/${declOrder}] Ablation B body line is not identical across contexts.`
        );
      }
    }
  }

  return rows;
}

function assertCellB(cellId, item, decl, bodyLine, row) {
  const { prefix } = row;
  const { targetVar, controlVar } = decl;

  if (!prefix.endsWith("[")) throw new Error(`[${cellId}] prefix must end exactly at "[".`);
  if (prefix.endsWith("[ ") || prefix.endsWith("[\n")) {
    throw new Error(`[${cellId}] prefix has trailing whitespace after "[".`);
  }
  if (targetVar === controlVar) throw new Error(`[${cellId}] target_var and control_var match.`);

  const declBlock = prefix.slice(0, row.decl_block_end_offset);
  const setter = "set" + capitalize(targetVar);
  if (!declBlock.includes(`const [${targetVar}, ${setter}] = useState(`)) {
    throw new Error(`[${cellId}] target (state-form) declaration is missing or not in scope.`);
  }
  if (!declBlock.includes(`const ${controlVar} = `)) {
    throw new Error(`[${cellId}] control (const-form) declaration is missing or not in scope.`);
  }
  if (!(mentions(bodyLine, targetVar) && mentions(bodyLine, controlVar))) {
    throw new Error(`[${cellId}] body must mention both target and control.`);
  }
}

function assertCrossingB(rows, pairCount) {
  const expectedTotal =
    pairCount * B_CONTEXT_ORDER.length * STATE_ROLES.length * DECL_ORDERS.length;
  if (rows.length !== expectedTotal) {
    throw new Error(`Ablation B: expected ${expectedTotal} rows, got ${rows.length}.`);
  }
  const byPair = groupBy(rows, (r) => r.pair_id);
  for (const [pairId, cells] of Object.entries(byPair)) {
    const expected = B_CONTEXT_ORDER.length * STATE_ROLES.length * DECL_ORDERS.length;
    if (cells.length !== expected) {
      throw new Error(`[${pairId}] Ablation B expected ${expected} cells, got ${cells.length}.`);
    }
    const combos = new Set(
      cells.map((r) => `${r.condition}|${r.state_role}|${r.decl_order}`)
    );
    if (combos.size !== expected) {
      throw new Error(`[${pairId}] Ablation B context/role/decl crossing has duplicates.`);
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

const rowsC = [];
const rowsB = [];
items.forEach((item, idx) => {
  const pairId = `pair_${String(idx + 1).padStart(2, "0")}`;
  rowsC.push(...buildRowsC(item, pairId));
  if (INCLUDE_B) rowsB.push(...buildRowsB(item, pairId));
});

assertDeclIdentityAcrossBodyuseC(rowsC);
assertCrossingC(rowsC, items.length);
if (INCLUDE_B) assertCrossingB(rowsB, items.length);

const rows = [...rowsC, ...rowsB];
fs.writeFileSync(OUTPUT, JSON.stringify(rows, null, 2), "utf-8");
console.log(
  `OK: ablation C = ${rowsC.length} rows` +
    (INCLUDE_B ? `, ablation B = ${rowsB.length} rows` : ", ablation B skipped (--no-b)") +
    ` -> ${rows.length} total prefixes -> ${OUTPUT}`
);

if (rowsC.length > 0) {
  const firstPair = rowsC[0].pair_id;
  console.log(`\nSample: ${firstPair}, ablation C, useEffect context`);
  for (const row of rowsC.filter(
    (r) => r.pair_id === firstPair && r.condition === "useEffect" && r.state_role === "dep_reactive" && r.decl_order === "reactive_first"
  )) {
    console.log(
      `\n[${row.id}] bodyuse=${row.bodyuse} body_order=${row.body_order} ` +
        `target=${row.target_var} control=${row.control_var}`
    );
    console.log(row.prefix);
  }
}

if (INCLUDE_B && rowsB.length > 0) {
  const firstPair = rowsB[0].pair_id;
  console.log(`\nSample: ${firstPair}, ablation B, dep_reactive/reactive_first across contexts`);
  for (const row of rowsB.filter(
    (r) => r.pair_id === firstPair && r.state_role === "dep_reactive" && r.decl_order === "reactive_first"
  )) {
    console.log(`\n[${row.id}] context=${row.condition}`);
    console.log(row.prefix);
  }
}
