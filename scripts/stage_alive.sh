#!/bin/bash
# Is a named fetch stage actually running? Sourced by the chain scripts.
#
# WHY THIS EXISTS: the obvious check, `pgrep -f <script name>`, is wrong in
# a way that deadlocked the fetch queue TWICE. `pgrep -f` matches the WHOLE
# command line, so it fires on any process that merely MENTIONS the script
# — including the session's own monitor loop, whose text greps for exactly
# these names. The waiter ends up waiting on the watcher, forever, and the
# failure is silent: a stalled queue looks identical to a slow one.
#
# Two narrower patterns were tried and both leaked:
#
#   1. Anchoring on 'scripts/<name>.sh' — defeated because a monitor that
#      stores that anchored pattern in a variable carries the anchored
#      string verbatim on its own command line.
#   2. Requiring a literal 'bash ' prefix — better, but it only held
#      because the monitor happened to store the pattern with a regex
#      escape ('intraday\.sh'), which is textually different from what the
#      regex matches. A monitor written without the escape would self-match
#      again. Correct by accident is not correct.
#
# THE FIX IS TO STOP SUBSTRING-MATCHING. A real invocation has the script
# as argv[1] with bash as argv[0]; a mere mention has it buried somewhere
# in argv[2..] of some other program. Comparing those two FIELDS instead of
# scanning the whole line makes the check immune to the entire class — no
# monitor, log line, or editor session can put the script path in argv[1]
# of a bash process without in fact being that script.
#
# Usage: stage_alive 'scripts/a.sh|scripts/b.sh'  -> exit 0 if any is live.
stage_alive() {
  ps -eo command= | awk -v pats="$1" '
    BEGIN { n = split(pats, want, "|") }
    $1 ~ /(^|\/)bash$/ {
      for (i = 1; i <= n; i++)
        if ($2 == want[i]) { found = 1; exit }
    }
    END { exit !found }
  '
}
