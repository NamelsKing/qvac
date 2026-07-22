# AGENTS.md

For the full build/test quick reference (per-package commands, native addon
builds, env tokens, CI layout), see `CLAUDE.md` at the repo root — it is the
canonical developer reference for this monorepo. This file only adds
Cursor-Cloud-specific, non-obvious operating notes.

## Cursor Cloud specific instructions

This is a package-scoped monorepo (there is **no root `package.json`** and no
workspace linking). Install/build/run inside each package you work on. The
startup update script installs the shared toolchain (`bun`, `bare`,
`bare-make`) and refreshes deps for the JS/TS surfaces below; it does **not**
build the C++ native addons.

### Scope that is set up and verified in Cloud

The Cloud environment targets the primary JS/TypeScript developer surfaces:

| Surface | Path | Pkg mgr | Build / run | Notes |
|---|---|---|---|---|
| SDK (canonical API) | `packages/sdk` | bun | `bun run build`, `bun run test:unit` | Build = lint + typecheck + tsc + alias resolve |
| CLI + OpenAI server | `packages/cli` | npm | `npm run build`, `node dist/index.js doctor`, `node dist/index.js serve openai` | Pulls the **published** `@qvac/sdk` from npm |
| Docs website | `docs/website` | npm | `npm run dev` (port **3001**) | Next.js 16; `dev` first compiles SDK examples |

Out of scope for the automatic setup (require clang-22 + vcpkg + `bare-make`,
and often GPU / mobile toolchains): the native C++ addon packages
(`packages/*-cpp`, `packages/*-ggml`, `packages/onnx`, …), the full SDK e2e
suite (needs an MQTT broker), the P2P `registry-server`, and mobile/Expo. Build
those manually per `CLAUDE.md` only when working on them.

### Non-obvious caveats

- **`@qvac/*` packages are public on npm** — no `NPM_TOKEN`/`.npmrc` is needed
  to install or run the SDK/CLI. `GH_TOKEN`/`HF_TOKEN` are only needed for
  native-addon vcpkg builds and model-license checks, not for the JS surfaces.
- **Inference works out of the box via published prebuilds.** The SDK spawns a
  **Bare** worker (over an IPC socket) that loads the `linux-x64` `.bare`
  prebuild shipped inside `node_modules/@qvac/llm-llamacpp/prebuilds/`. Models
  download over the public P2P registry (Hyperswarm, no token) and are cached
  under `~/.qvac/models`.
- **CPU-only.** No Vulkan ICD is present in the VM, so inference runs on CPU
  (slower). `qvac doctor` reports this as a warning — it is expected, not a
  failure. `qvac doctor` (and mobile toolchain warnings for adb/Xcode) exit 0.
- **CLI ↔ in-repo SDK.** A plain `npm install` in `packages/cli` uses the
  *published* SDK. To iterate the CLI against the *in-repo* SDK, use
  `npm run dev:link` (or `sdk-source:workspace`) and run `npm run dev:unlink`
  before committing so the committed `@qvac/sdk` range is restored (see
  `packages/cli/README.md`).
- **Two SDK unit tests fail deterministically in this VM** and are unrelated to
  setup: `test/unit/worker-crash-loadmodel.test.ts` and
  `worker-crash-doomed.test.ts`. The Bare worker spawns and the call rejects,
  but the error is classified as `RPCError` instead of the expected
  `WORKER_CRASHED` after a `SIGKILL`. All other unit tests pass.
- **`bun install` prints "Blocked 1 postinstall" for `unrs-resolver`** — safe
  to ignore. Its native binding is installed as an optional dependency, so
  `bun run lint` works without trusting the postinstall.
- **Toolchain paths.** `bun` lives in `~/.bun/bin`; global npm binaries
  (`bare`, `bare-make`) use a user-writable prefix at `~/.npm-global/bin`. Both
  are added to `~/.bashrc`. The `nvm ... has a prefix setting incompatible with
  nvm` warning during `npm install` is cosmetic and can be ignored.
- **Default ports:** CLI `serve openai` → `11434`; docs dev → `3001`.
