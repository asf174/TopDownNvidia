import re
file = """==PROF== Connected to process 60621 (/afs/atc.unican.es/u/a/alvaro/TopDownNvidia/CUDA/bin/add_two_matrix)
==PROF== Profiling "addMatrix" - 1: 0%....50%....100% - 1 pass
==PROF== Profiling "addMatrix2" - 2: 0%....50%....100% - 1 pass
NUMBLOCKS: 47852 THREADS_PER_BLOCK: 256
==PROF== Disconnected from process 60621
[60621] add_two_matrix@127.0.0.1
  addMatrix(int*, int*, int*, int), 2021-Mar-29 21:11:57, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__warp_issue_stalled_drain_per_warp_active.pct                                   %                           2,75
    smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct                             %                           0,15
    ---------------------------------------------------------------------- --------------- ------------------------------

  addMatrix2(int*, int*, int*, int), 2021-Mar-29 21:11:57, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    smsp__warp_issue_stalled_drain_per_warp_active.pct                                   %                           2,76
    smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct                             %                           0,13
    ---------------------------------------------------------------------- --------------- ------------------------------"""
for line in file.splitlines():
    line = re.sub(' +', ' ', line) # delete more than one spaces and put only one
    list_words = line.split(" ")
    if len(list_words) == 4 and list_words[1][0] != "-":
        name = list_words[1]
        unit = list_words[2]
        value = list_words[3]
        print(name + " " + unit + " " + value)