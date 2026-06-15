/**
 * generate_dataset.js
 * variable_names_validated.json을 읽어 React / Generic JS 템플릿에 통합합니다.
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

/** "flag" → "Flag" (setter 이름용 파스칼케이스 첫 글자) */
function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

/** "UserCard" → "userCard" (카멜케이스) */
function toCamelCase(word) {
  return word.charAt(0).toLowerCase() + word.slice(1);
}

// -------------------------------------------------------
// 템플릿 생성 함수
// -------------------------------------------------------

function makeReactClean({ Component, dep, alt, init, param }) {
  const Dep = capitalize(dep);
  return [
    `function ${Component}() {`,
    `  const [${dep}, set${Dep}] = useState(${init});`,
    `  const ${alt} = ${init};`,
    ``,
    `  useEffect(() => {`,
    `    fetch(\`/api?${param}=\${${dep}}\`);`,
    `  }, [${dep}])`,
    `}`,
  ].join("\n");
}

function makeReactCorrupted({ Component, dep, alt, init, param }) {
  const Dep = capitalize(dep);
  return [
    `function ${Component}() {`,
    `  const [${dep}, set${Dep}] = useState(${init});`,
    `  const ${alt} = ${init};`,
    ``,
    `  useEffect(() => {`,
    `    fetch(\`/api?${param}=\${${alt}}\`);`,
    `  }, [${dep}])`,
    `}`,
  ].join("\n");
}

function makeJsClean({ Component, dep, alt, init, param }) {
  const Dep = capitalize(dep);
  const component = toCamelCase(Component);
  return [
    `function subscribe(callback, deps) { deps.forEach(d => callback(d)); }`,
    ``,
    `function ${component}() {`,
    `  const [${dep}, set${Dep}] = useState(${init});`,
    `  const ${alt} = ${init};`,
    ``,
    `  subscribe(() => {`,
    `    fetch(\`/api?${param}=\${${dep}}\`);`,
    `  }, [${dep}])`,
    `}`,
  ].join("\n");
}

function makeJsCorrupted({ Component, dep, alt, init, param }) {
  const Dep = capitalize(dep);
  const component = toCamelCase(Component);
  return [
    `function subscribe(callback, deps) { deps.forEach(d => callback(d)); }`,
    ``,
    `function ${component}() {`,
    `  const [${dep}, set${Dep}] = useState(${init});`,
    `  const ${alt} = ${init};`,
    ``,
    `  subscribe(() => {`,
    `    fetch(\`/api?${param}=\${${alt}}\`);`,
    `  }, [${dep}])`,
    `}`,
  ].join("\n");
}

function makeDescription({ dep, alt, param }) {
  return `${param} fetch — body: ${dep}→${alt}, array stays [${dep}]`;
}

// -------------------------------------------------------
// 메인
// -------------------------------------------------------

const raw = fs.readFileSync(INPUT, "utf-8");
const items = JSON.parse(raw);

const dataset = items.map((item, idx) => {
  const id = `sub_${String(idx + 1).padStart(2, "0")}`;
  const { Component, dep, alt, init, param } = item;

  // _token_forms에 공백 포함 여부가 기록되어 있으면 그걸 사용
  // 없으면 공백 포함 형태를 기본값으로 사용 (TransformerLens 기본)
  const forms = item._token_forms || {};
  const clean_target = forms.dep ?? ` ${dep}`;
  const corrupted_target = forms.alt ?? ` ${alt}`;

  return {
    id,
    type: "substitution",
    description: makeDescription({ dep, alt, param }),
    react_clean: makeReactClean({ Component, dep, alt, init, param }),
    react_corrupted: makeReactCorrupted({ Component, dep, alt, init, param }),
    js_clean: makeJsClean({ Component, dep, alt, init, param }),
    js_corrupted: makeJsCorrupted({ Component, dep, alt, init, param }),
    clean_target,
    corrupted_target,
  };
});

fs.writeFileSync(OUTPUT, JSON.stringify(dataset, null, 2), "utf-8");
console.log(`✓ ${dataset.length}개 항목 생성 → ${OUTPUT}`);

// 샘플 출력 (첫 번째 항목)
console.log("\n--- 샘플 (첫 번째 항목) ---");
const sample = dataset[0];
console.log(`id: ${sample.id}`);
console.log(`description: ${sample.description}`);
console.log(`clean_target: "${sample.clean_target}"`);
console.log(`corrupted_target: "${sample.corrupted_target}"`);
console.log(`\n[react_clean]\n${sample.react_clean}`);
console.log(`\n[react_corrupted]\n${sample.react_corrupted}`);
console.log(`\n[js_clean]\n${sample.js_clean}`);
console.log(`\n[js_corrupted]\n${sample.js_corrupted}`);