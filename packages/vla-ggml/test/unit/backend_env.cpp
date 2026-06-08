// Global test environment: load the ggml backends once before any test runs.
//
// In production the addon loads its backends at init via
// vla_backend_selection::loadBackendsOnce() (called from the JS binding). The
// unit tests link the addon objects directly and bypass that path, so under
// GGML_BACKEND_DL=ON no backend — not even CPU — is registered. Any test that
// computes a graph or constructs a model then fails with "no CPU backend
// available". Registering this environment loads the backends the same way the
// addon does, before the first test, so the suite mirrors production.

#include <gtest/gtest.h>

#include "utils/BackendSelection.hpp"

namespace {

class BackendEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { vla_backend_selection::loadBackendsOnce(""); }
};

// Registered at static-init (before main), so gtest_main runs SetUp() ahead of
// RUN_ALL_TESTS. The returned pointer is owned by gtest.
const ::testing::Environment* const kBackendEnvironment =
    ::testing::AddGlobalTestEnvironment(new BackendEnvironment);

}  // namespace
