# Scripted-client correction before the second attempt

The first run at `e5142b7` failed all four harness verdicts. Preserve it unchanged.
It made no model calls. Its observations correct assumptions in the original
protocol and its test implementation:

- Fresh isolated configuration did not exercise proactive compaction in the auto
  case. Name an auto-compact window explicitly in both test cases using the accepted
  100000 minimum; the installed implementation caps it at the truthful 18432 model
  window. This is a test-only path selection, not an increased context declaration
  or a change to the production launcher's window.
- Repeated reads of one unchanged file return a cached-read reminder. Use five
  distinct small files so five successful content-bearing reads can be required.
  Never count a denied or missing result as a successful read.
- The installed Read implementation trims the oversized file: metadata reported
  `truncatedByTokenCap=true`, 168 returned lines out of 1501. Accept explicit cap
  trimming or rejection, followed by a successful two-line targeted read. Do not
  describe trimming as rejection or treat it as the complete file contents.
- `/compact` was unavailable under the existing command-disabling flag. Remove
  `--disable-slash-commands` and test the command again. Bare mode still suppresses
  discovery, and restricted mode, explicit tool selection and strict MCP remain.

All original safety, privacy, timeout and provenance requirements still apply.
Run the same four cases in a fresh directory after review and commit. These checks
exercise client control flow with scripted responses, not live-model summaries or
an interactive terminal rendering qualification.
