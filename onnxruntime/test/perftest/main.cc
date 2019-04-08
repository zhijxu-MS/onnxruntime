// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/common/logging/sinks/clog_sink.h>
#include <core/common/logging/logging.h>
#include <core/framework/environment.h>
#include <core/platform/env.h>

#include "command_args_parser.h"
#include "performance_runner.h"

using namespace onnxruntime;

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[], OrtEnv** p_env) {
#else
int real_main(int argc, char* argv[], OrtEnv** p_env) {
#endif
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    perftest::CommandLineParser::ShowUsage();
    return -1;
  }
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  OrtEnv* env;
  {
    OrtStatus* ost = OrtCreateEnv(logging_level, "Default", &env);
    if (ost != nullptr) {
      fprintf(stderr, "Error creating environment: %s \n", OrtGetErrorMessage(ost));
      OrtReleaseStatus(ost);
      return -1;
    }
    *p_env = env;
  }
  perftest::PerformanceRunner perf_runner(env, test_config);
  auto status = perf_runner.Run();
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "Run failed:%s", status.ErrorMessage().c_str());
    return -1;
  }

  perf_runner.SerializeResult();

  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  OrtEnv* env = nullptr;
  int retval = -1;
  try {
    retval = real_main(argc, argv, &env);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }
  if (env) {
    OrtReleaseEnv(env);
  } else {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
  return retval;
}
