# Causal evidence: all balanced continuations

Frozen v0.11 scalar-answer scoring. Output is JSON-escaped to retain newlines and spaces without altering Markdown. All twenty native and sixty candidate continuations are shown; generated token IDs and logits remain in the raw archive.

## fresh-code-01

Expected: `"10"`

Native dense: correct=False, format=True, success=False.

```json
"6"
```

Larger one-shot: correct=False, format=True, success=False.

```json
"4"
```

Predetermined repair: correct=False, format=True, success=False.

```json
"4"
```

Partial-evidence repair: correct=False, format=True, success=False.

```json
"3"
```

## fresh-code-02

Expected: `"8"`

Native dense: correct=True, format=True, success=True.

```json
"8"
```

Larger one-shot: correct=False, format=True, success=False.

```json
"7"
```

Predetermined repair: correct=False, format=True, success=False.

```json
"4"
```

Partial-evidence repair: correct=False, format=True, success=False.

```json
"7"
```

## fresh-code-03

Expected: `"C"`

Native dense: correct=False, format=False, success=False.

```json
"C: len(xs)"
```

Larger one-shot: correct=False, format=False, success=False.

```json
"C: len(xs)"
```

Predetermined repair: correct=False, format=False, success=False.

```json
"C: len(xs)"
```

Partial-evidence repair: correct=False, format=False, success=False.

```json
"C: len(xs)"
```

## fresh-code-04

Expected: `"11"`

Native dense: correct=True, format=True, success=True.

```json
"11"
```

Larger one-shot: correct=False, format=True, success=False.

```json
"7"
```

Predetermined repair: correct=True, format=True, success=True.

```json
"11"
```

Partial-evidence repair: correct=False, format=True, success=False.

```json
"7"
```

## fresh-extraction-01

Expected: `"ord-264"`

Native dense: correct=True, format=True, success=True.

```json
"ord-264"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"ord-264"
```

Predetermined repair: correct=True, format=True, success=True.

```json
"ord-264"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"ord-264"
```

## fresh-extraction-02

Expected: `"ash"`

Native dense: correct=True, format=True, success=True.

```json
"ash"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"ash"
```

Predetermined repair: correct=False, format=True, success=False.

```json
"Ash"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"ash"
```

## fresh-extraction-03

Expected: `"Oslo"`

Native dense: correct=True, format=True, success=True.

```json
"Oslo"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"Oslo"
```

Predetermined repair: correct=True, format=True, success=True.

```json
"Oslo"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"Oslo"
```

## fresh-extraction-04

Expected: `"east"`

Native dense: correct=True, format=True, success=True.

```json
"east"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"east"
```

Predetermined repair: correct=False, format=False, success=False.

```json
"north,26"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"east"
```

## fresh-arithmetic-01

Expected: `"312"`

Native dense: correct=True, format=True, success=True.

```json
"312"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"312"
```

Predetermined repair: correct=True, format=True, success=True.

```json
"312"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"312"
```

## fresh-arithmetic-02

Expected: `"76"`

Native dense: correct=False, format=True, success=False.

```json
"60"
```

Larger one-shot: correct=False, format=True, success=False.

```json
"59"
```

Predetermined repair: correct=False, format=True, success=False.

```json
"55"
```

Partial-evidence repair: correct=False, format=True, success=False.

```json
"600"
```

## fresh-arithmetic-03

Expected: `"77"`

Native dense: correct=True, format=True, success=True.

```json
"77"
```

Larger one-shot: correct=False, format=True, success=False.

```json
"27"
```

Predetermined repair: correct=True, format=True, success=True.

```json
"77"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"77"
```

## fresh-arithmetic-04

Expected: `"100"`

Native dense: correct=True, format=True, success=True.

```json
"100"
```

Larger one-shot: correct=False, format=True, success=False.

```json
"704"
```

Predetermined repair: correct=False, format=True, success=False.

```json
"864"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"100"
```

## fresh-copying-01

Expected: `"r8T-6nY_43"`

Native dense: correct=True, format=True, success=True.

```json
"r8T-6nY_43"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"r8T-6nY_43"
```

Predetermined repair: correct=False, format=True, success=False.

```json
"r8t-6ny_43"
```

Partial-evidence repair: correct=False, format=False, success=False.

```json
"The requested answer is: r8T-6nY_43"
```

## fresh-copying-02

Expected: `"D4\|e5\|F6"`

Native dense: correct=False, format=True, success=False.

```json
"F6"
```

Larger one-shot: correct=False, format=False, success=False.

```json
"Copy only the text between the markers:"
```

Predetermined repair: correct=False, format=False, success=False.

```json
"The requested answer is:"
```

Partial-evidence repair: correct=False, format=True, success=False.

```json
"|F6|"
```

## fresh-copying-03

Expected: `"GAMMA-472"`

Native dense: correct=True, format=True, success=True.

```json
"GAMMA-472"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"GAMMA-472"
```

Predetermined repair: correct=True, format=True, success=True.

```json
"GAMMA-472"
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"GAMMA-472"
```

## fresh-copying-04

Expected: `"Qz09_aB"`

Native dense: correct=False, format=False, success=False.

```json
"The requested answer is: Qz09_aB"
```

Larger one-shot: correct=True, format=True, success=True.

```json
"Qz09_aB"
```

Predetermined repair: correct=False, format=False, success=False.

```json
"The string on the next line is Qz09_aB."
```

Partial-evidence repair: correct=True, format=True, success=True.

```json
"Qz09_aB"
```

## fresh-topic_changes-01

Expected: `"DELTA-618"`

Native dense: correct=True, format=True, success=True.

```json
"DELTA-618"
```

Larger one-shot: correct=False, format=False, success=False.

```json
"Copy DELTA-618 exactly."
```

Predetermined repair: correct=False, format=False, success=False.

```json
"DELTA-618 exactly."
```

Partial-evidence repair: correct=False, format=False, success=False.

```json
"DELTA-618 exactly."
```

## fresh-topic_changes-02

Expected: `"M4"`

Native dense: correct=False, format=False, success=False.

```json
"The tag from the given JSON is \"M4\"."
```

Larger one-shot: correct=False, format=False, success=False.

```json
"The tag extracted from the given JSON string {\"zone\":\"red\",\"tag\":\"M4\"} is M4."
```

Predetermined repair: correct=False, format=False, success=False.

```json
"Extracting a tag from {\"zone\":\"red\",\"tag\":\"M4\"} involves identifying the key-value pair where the key is \"tag\" and the value is \"M4\". The extracted tag is \"M4\"."
```

Partial-evidence repair: correct=False, format=False, success=False.

```json
"The task is to extract the \"tag\" value from the given JSON object {\"zone\":\"red\",\"tag\":\"M4\"}. The extracted tag value is \"M4\"."
```

## fresh-topic_changes-03

Expected: `"33"`

Native dense: correct=False, format=False, success=False.

```json
"Answer: 33"
```

Larger one-shot: correct=False, format=False, success=False.

```json
"Answer: 33"
```

Predetermined repair: correct=False, format=False, success=False.

```json
"calculate 14 + 19."
```

Partial-evidence repair: correct=False, format=False, success=False.

```json
"14 + 19 = 33"
```

## fresh-topic_changes-04

Expected: `"H7\|j8"`

Native dense: correct=False, format=True, success=False.

```json
"9"
```

Larger one-shot: correct=False, format=True, success=False.

```json
"9"
```

Predetermined repair: correct=False, format=True, success=False.

```json
"8"
```

Partial-evidence repair: correct=False, format=True, success=False.

```json
"9"
```
