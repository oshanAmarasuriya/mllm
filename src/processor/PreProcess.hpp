//
// Created by 咸的鱼 on 2023/12/21.
//

#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP
#include "tokenizers/Tokenizer.hpp"
namespace mllm {
class PreProcessor {
protected:
    mllm::Tokenizer *tokenizer_;
public:
    explicit PreProcessor(mllm::Tokenizer *tokenizer) : tokenizer_(tokenizer) {}
    virtual void PreProcessImages(const std::vector<uint8_t*> &images,const std::vector<size_t> &image_length, int height = 1080, int width = 1920, bool do_pad = true, bool do_resize = true, bool do_normalize = true, float mean = 0.5, float std = 0.5);
    virtual void Process(const std::string& text);
    virtual void PreProcessImages(const std::vector<std::string> &images_path,int height = 1080,int width = 1920, bool do_pad = true, bool do_resize = true, bool do_normalize = true, float mean = 0.5, float std = 0.5);

};
}

#endif //PREPROCESS_HPP
