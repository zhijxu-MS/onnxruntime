#include "core/common/status.h"
#include "core/common/common.h"

namespace Lotus {
namespace Common {
Status::Status(StatusCategory category, int code, const std::string& msg) {
  state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code)
    : Status(category, code, EmptyString()) {
}

bool Status::IsOK() const noexcept {
  return (state_ == NULL);
}

StatusCategory Status::Category() const noexcept {
  return IsOK() ? StatusCategory::NONE : state_->category;
}

int Status::Code() const noexcept {
  return IsOK() ? static_cast<int>(StatusCode::OK) : state_->code;
}

const std::string& Status::ErrorMessage() const {
  return IsOK() ? EmptyString() : state_->msg;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return std::string("OK");
  }

  std::string result;

  if (StatusCategory::SYSTEM == state_->category) {
    result += "SystemError";
    result += " : ";
    result += std::to_string(errno);
  } else if (StatusCategory::LOTUS == state_->category) {
    result += "[LotusError]";
    result += " : ";
    result += std::to_string(Code());
    std::string msg;

    switch (static_cast<StatusCode>(Code())) {
      case INVALID_ARGUMENT:
        msg = "INVALID_ARGUMENT";
        break;
      case NO_SUCHFILE:
        msg = "NO_SUCHFILE";
        break;
      case NO_MODEL:
        msg = "NO_MODEL";
        break;
      case ENGINE_ERROR:
        msg = "ENGINE_ERROR";
        break;
      case RUNTIME_EXCEPTION:
        msg = "RUNTIME_EXCEPTION";
        break;
      case INVALID_PROTOBUF:
        msg = "INVALID_PROTOBUF";
        break;
      case MODEL_LOADED:
        msg = "MODEL_LOADED";
        break;
      case NOT_IMPLEMENTED:
        msg = "NOT_IMPLEMENTED";
        break;
      case INVALID_GRAPH:
        msg = "INVALID_GRAPH";
        break;
      default:
        msg = "GENERAL ERROR";
        break;
    }
    result += " : ";
    result += msg;
    result += " : ";
    result += state_->msg;
  }

  return result;
}

const Status& Status::OK() noexcept {
  // We use 'new' to avoid static initialization issues
  //   https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
  // Suppressing r.11 due to that
  //   Warning C26409 Avoid calling new and delete explicitly, use std::make_unique<T> instead
  //   r.11: http://go.microsoft.com/fwlink/?linkid=845485
  GSL_SUPPRESS(r .11) {
    static Status* s_ok = new Status();
    return *s_ok;
  }
}

const std::string& Status::EmptyString() {
  GSL_SUPPRESS(r .11) {
    static std::string* s_empty = new std::string();
    return *s_empty;
  }
}
}  // namespace Common
}  // namespace Lotus
