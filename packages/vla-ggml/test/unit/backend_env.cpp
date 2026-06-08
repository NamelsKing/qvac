// Global test environment: load the ggml backends once before any test runs.
//
// In production the addon loads its backends at init via
// vla_backend_selection::loadBackendsOnce(backendsDir), where backendsDir is
// the prebuilds folder holding the DL backend modules. The unit tests link the
// addon objects directly and the factory test passes backendsDir="" — fine in a
// static build (CPU is linked in), but under GGML_BACKEND_DL=ON it means no
// backend gets loaded, so pi05LoadModel throws "no CPU backend available".
//
// The DL modules (libqvac-ggml-cpu-*.so) are installed next to the ggml core
// library this test binary links. Resolve that directory at runtime via
// dladdr() and hand it to loadBackendsOnce() — the same code path production
// uses, just with the test's own lib dir. loadBackendsOnce is std::call_once,
// so this first load wins and the factory test's later loadBackendsOnce("")
// becomes a no-op.

#include <dlfcn.h>

#include <filesystem>
#include <iostream>
#include <string>

#include <ggml-backend.h>
#include <gtest/gtest.h>

#include "utils/BackendSelection.hpp"

namespace {

// Directory of the ggml core library linked into this test binary, where the
// DL backend modules are co-installed. Empty string if it can't be resolved
// (e.g. a static build), which loadBackendsOnce treats as the default search.
std::string ggmlLibDir() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&ggml_backend_load_all), &info) != 0 &&
      info.dli_fname != nullptr) {
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }
  return "";
}

class BackendEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    const std::string dir = ggmlLibDir();
    std::cerr << "[backend_env] ggmlLibDir='" << dir << "'\n";
    std::error_code ec;
    if (!dir.empty() && std::filesystem::is_directory(dir, ec)) {
      for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.find("ggml") != std::string::npos &&
            n.find(".so") != std::string::npos) {
          std::cerr << "[backend_env]   module: " << n << "\n";
        }
      }
    }
    vla_backend_selection::loadBackendsOnce(dir);
    const size_t n = ggml_backend_dev_count();
    std::cerr << "[backend_env] ggml_backend_dev_count=" << n << "\n";
    for (size_t i = 0; i < n; ++i) {
      ggml_backend_dev_t d = ggml_backend_dev_get(i);
      std::cerr << "[backend_env]   dev[" << i
                << "] type=" << static_cast<int>(ggml_backend_dev_type(d))
                << " name=" << ggml_backend_dev_name(d) << "\n";
    }
  }
};

// Registered at static-init (before main), so gtest_main runs SetUp() ahead of
// RUN_ALL_TESTS. The returned pointer is owned by gtest.
const ::testing::Environment* const kBackendEnvironment =
    ::testing::AddGlobalTestEnvironment(new BackendEnvironment);

}  // namespace
