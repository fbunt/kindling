# TODO

## Security
- [x] Block write-method names in AST validation
- [ ] Sandbox query execution with read-only filesystem (bwrap or landlock)
  - Refactor `execute_query` to run in subprocess instead of thread
  - Use bubblewrap (`bwrap`) to bind-mount filesystem read-only
  - Or use landlock for kernel-native restrictions (no extra deps)
  - Designate writable tmpdir for plot output, copy results back
  - Eliminates zombie thread problem as a bonus
