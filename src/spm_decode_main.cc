// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include <functional>
#include <string>
#include <vector>
#include <variant>
#include <iostream>
#include <fstream>

#include "common.h"
#include "filesystem.h"
#include "init.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/str_split.h"
#include "util.h"

ABSL_FLAG(std::string, model, "", "model file name");
ABSL_FLAG(std::string, input, "", "input filename");
ABSL_FLAG(std::string, output, "", "output filename");
ABSL_FLAG(std::string, input_format, "piece", "choose from piece or id");
ABSL_FLAG(std::string, output_format, "string", "choose from string or proto");
ABSL_FLAG(std::string, extra_options, "",
          "':' separated encoder extra options, e.g., \"reverse:bos:eos\"");

std::vector<int> readFileToVector(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary | std::ios::in);

    if (!inFile) {
        std::cerr << "Error: Unable to open the file: " << filename << std::endl;
        return {};
    }

    // Move the file position pointer to the end of the file
    inFile.seekg(0, std::ios::end);

    // Calculate the number of elements in the file
    std::streamsize numElements = inFile.tellg() / sizeof(uint16_t);

    // Move the file position pointer back to the beginning of the file
    inFile.seekg(0, std::ios::beg);

    std::vector<int> result(numElements);

    for (std::size_t i = 0; i < numElements; ++i) {
        uint16_t num16;
        inFile.read(reinterpret_cast<char*>(&num16), sizeof(uint16_t));
        result[i] = static_cast<int>(num16);
    }

    inFile.close();
    return result;
}

int main(int argc, char *argv[]) {
  sentencepiece::ScopedResourceDestructor cleaner;
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);
  std::vector<std::string> rest_args;

  if (absl::GetFlag(FLAGS_input).empty()) {
    for (int i = 1; i < argc; ++i) {
      rest_args.push_back(std::string(argv[i]));
    }
  } else {
    rest_args.push_back(absl::GetFlag(FLAGS_input));
  }

  if (rest_args.empty())
    rest_args.push_back("");  // empty means that reading from stdin.

  CHECK(!absl::GetFlag(FLAGS_model).empty());

  sentencepiece::SentencePieceProcessor sp;
  CHECK_OK(sp.Load(absl::GetFlag(FLAGS_model)));
  CHECK_OK(sp.SetDecodeExtraOptions(absl::GetFlag(FLAGS_extra_options)));

  auto output =
      sentencepiece::filesystem::NewWritableFile(absl::GetFlag(FLAGS_output));
  CHECK_OK(output->status());

  std::string detok, line;
  sentencepiece::SentencePieceText spt;
  std::function<void(const std::vector<std::string> &pieces)> process;
  std::function<void(const std::vector<int> &pieces)> process_int;

  auto ToIds = [&](const std::vector<std::string> &pieces) {
    std::vector<int> ids;
    ids.reserve(pieces.size());
    for (const auto &s : pieces) {
      ids.push_back(atoi(s.c_str()));
    }
    return ids;
  };

  if (absl::GetFlag(FLAGS_input_format) == "piece") {
    if (absl::GetFlag(FLAGS_output_format) == "string") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(pieces, &detok));
        output->WriteLine(detok);
      };
    } else if (absl::GetFlag(FLAGS_output_format) == "proto") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(pieces, &spt));
      };
    } else {
      LOG(FATAL) << "Unknown output format: "
                 << absl::GetFlag(FLAGS_output_format);
    }
  } else if (absl::GetFlag(FLAGS_input_format) == "map") {
    if (absl::GetFlag(FLAGS_output_format) == "string") {
      process_int = [&](const std::vector<int> &ids) {
        CHECK_OK(sp.Decode(ids, &detok));
        output->WriteLine(detok);
      };
    } else if (absl::GetFlag(FLAGS_output_format) == "proto") {
      process_int = [&](const std::vector<int> &ids) {
        CHECK_OK(sp.Decode(ids, &spt));
      };
    } else {
      LOG(FATAL) << "Unknown output format: "
                 << absl::GetFlag(FLAGS_output_format);
    }
  } else if (absl::GetFlag(FLAGS_input_format) == "id") {
    if (absl::GetFlag(FLAGS_output_format) == "string") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(ToIds(pieces), &detok));
        output->WriteLine(detok);
      };
    } else if (absl::GetFlag(FLAGS_output_format) == "proto") {
      process = [&](const std::vector<std::string> &pieces) {
        CHECK_OK(sp.Decode(ToIds(pieces), &spt));
      };
    } else {
      LOG(FATAL) << "Unknown output format: "
                 << absl::GetFlag(FLAGS_output_format);
    }
  } else {
    LOG(FATAL) << "Unknown input format: " << absl::GetFlag(FLAGS_input_format);
  }

  if (absl::GetFlag(FLAGS_input_format) == "map") {
    for (const auto &filename : rest_args) {
      auto ids = readFileToVector(filename);
      process_int(ids);
    }
  } else {
    for (const auto &filename : rest_args) {
      auto input = sentencepiece::filesystem::NewReadableFile(filename);
      CHECK_OK(input->status());
      while (input->ReadLine(&line)) {
        const auto pieces = absl::StrSplit(line, " ");
        process(pieces);
      }
    }
  }

  return 0;
}
