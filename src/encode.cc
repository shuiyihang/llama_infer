#include "encode.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "absl/strings/str_replace.h"
#include "nlohmann/json.hpp"
#include "tiktoken.h"
#include "unicode.h"
#include "unordered_dense.h"

static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeEncodeLayer::BpeEncodeLayer(std::string tokenizer_pth) : EncodeLayerBase(std::move(tokenizer_pth)) {
  using json = nlohmann::json;
  std::ifstream f(m_tokenizer_pth);
  json data = json::parse(f);
  const auto &datas = data["added_tokens"];
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  for (const auto &data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  const auto &vocabs = data["model"]["vocab"];
  const auto &vocab_items = vocabs.items();
  for (const auto &v : vocab_items) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    encoder[key] = id;
  }
  m_eog_tokens.emplace(special_tokens["<|im_end|>"]);
  m_eog_tokens.emplace(special_tokens["<|endoftext|>"]);

  m_num_tokens = encoder.size() + special_tokens.size();
  m_tiktoken = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string &sentence) const {
  std::map<std::string, std::string> replacements;
  replacements[" "] = "Ġ";
  std::string ss = absl::StrReplaceAll(sentence, replacements);
  std::vector<int32_t> tokens = m_tiktoken->encode(ss);

  return tokens;
}
std::string BpeEncodeLayer::decode(std::vector<int32_t> &token) const {
  std::string ss = m_tiktoken->decode(token);
  std::map<std::string, std::string> replacements;
  replacements["Ġ"] = " ";
  std::string sentence = absl::StrReplaceAll(ss, replacements);
  return sentence;
}
int32_t BpeEncodeLayer::vocab_size() const { return m_num_tokens; }

bool BpeEncodeLayer::is_sentence_ending(int32_t token) {
  if (m_eog_tokens.count(token) != 0) return true;
  return false;
}