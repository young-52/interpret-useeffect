You are generating variable name sets for a React useEffect dataset.

## Template

```jsx
import { useEffect, useState } from 'react';
function {Component}() {
  const [{dep}, set{Dep}] = useState({init});
  const {alt} = {init};
  useEffect(() => {
    fetch(`/api?{param}=${{{dep}}}`);
  }, [{dep}])
}
```

The corrupted version replaces every occurrence of {dep} in the effect body with {alt}.
The dependency array stays [{dep}] in both versions.

## Task

Generate exactly 50 unique tuples of (Component, dep, alt, init, param).

## Rules
### Component

- PascalCase
- Must tokenize as a single token in the Llama-3.2-1B tokenizer
- Safe examples (confirmed single-token): `UserCard`, `Dashboard`, `SearchBox`, `Profile`, `Settings`
- Avoid long multi-word compounds likely to split (e.g. `ProductListPage`, `AdminDashboard`)

### dep

- camelCase, single token
- Represents a React state variable — should read naturally as a piece of UI state
- Safe examples (confirmed single-token): `name`, `count`, `page`, `query`, `title`,
  `color`, `limit`, `mode`, `index`, `status`, `id`, `key`, `flag`, `text`, `tag`
- Avoid: `userId`, `pageSize`, `isLoading` — likely to split into multiple tokens

### alt
- camelCase, single token
- Represents a plain local const variable (not state) — same semantic domain as dep
  but clearly a non-state name
- Must be different from dep
- Safe examples: `base`, `ref`, `size`, `max`, `min`, `cap`, `len`, `num`, `val`, `offset`
- Avoid repeating dep as alt

### init

- Integer literal or boolean literal, single token
- Integer examples: `0`, `1`, `2`, `10`, `100`
- Boolean examples: `true`, `false`

### param

- camelCase, single token
- Must be semantically coherent with init:
  - If init is an integer → param should read as a numeric identifier
    (e.g. `page`, `count`, `id`, `index`, `limit`, `size`, `num`)
  - If init is a boolean → param should read as a flag or toggle
    (e.g. `flag`, `active`, `enabled`, `visible`, `open`, `show`)
- Must be different from dep and alt

## Output format

Return a JSON array of exactly 50 objects. No explanation, no markdown, no preamble.
[
  {
    "Component": "UserCard",
    "dep": "count",
    "alt": "base",
    "init": "0",
    "param": "page"
  },
  ...
]
