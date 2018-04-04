#include "core/framework/init.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace Lotus {

using namespace Lotus::Common;

Status Initializer::EnsureInitialized(int* pargc, char*** pargv) {
  static Initializer initializer{pargc, pargv};
  return initializer.initialization_status_;
}

Status Initializer::EnsureInitialized() {
  int argc = 0;
  char* argv_buf[] = {nullptr};
  char** argv = argv_buf;
  return EnsureInitialized(&argc, &argv);
}

Initializer::Initializer(int* pargc, char*** pargv)
    : initialization_status_{Initialize(pargc, pargv)} {
}

Status Initializer::Initialize(int* pargc, char*** pargv) {
  try {
    Status status{};

    if (!pargc || !pargv) status = Status(LOTUS, StatusCode::INVALID_ARGUMENT);
    if (!status.IsOK()) return status;

    // Register microsoft domain with min/max op_set version as 1/1.
    onnx::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(LotusIR::kMSDomain, 1, 1);
    // Register microsoft domain ops.
    RETURN_IF_ERROR(LotusIR::MsOpRegistry::RegisterMsOps());

    // LotusDeviceManager
    auto& allocator_manager = AllocatorManager::Instance();

    status = allocator_manager.InitializeAllocators();
    if (status.IsOK())
      return status;

    return Status::OK();
  } catch (std::exception& ex) {
    return Status{LOTUS, StatusCode::RUNTIME_EXCEPTION,
                  std::string{"Exception caught: "} + ex.what()};
  } catch (...) {
    return Status{LOTUS, StatusCode::RUNTIME_EXCEPTION};
  }
}
}  // namespace Lotus
