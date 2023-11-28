//
// Created by 咸的鱼 on 2023/11/26.
//

#ifndef MLLM_CPULAYERNORM_HPP
#define MLLM_CPULAYERNORM_HPP

#include "CPUBackend.hpp"
namespace mllm {

class CPULayerNorm:public Op {
public:
    CPULayerNorm(Backend *bn, string opName, bool multiThread, float epsilon = 1e-5, bool bias= true);
    virtual ~CPULayerNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;

private:
    bool support_multi_thread_ = false;
    float epsilon_ = 1e-5;
    int normSize_=0;
    Tensor weight_;
    Tensor bias_;
    bool  bias;
};
class CPULayerNormCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {

        return new CPULayerNorm(bn, name, false, true);
    }
};


} // namespace mllm

#endif // MLLM_CPULAYERNORM_HPP
