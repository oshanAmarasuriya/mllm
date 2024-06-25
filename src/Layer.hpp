//
// Created by Rongjie Yi on 2024/1/29 0029.
//

#ifndef OPERATION_H
#define OPERATION_H

#include <utility>

#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"

#include <Module.hpp>

#include <regex>
#include <string>
#include <vector>

namespace mllm {

class Layer {
public:
    Layer() = default;
    void init(std::string name, OpType type) {
        name_ = std::move(name);
        param_["type"] = type;// eg:- for embedding purpose, type is EMBEDDING
        Module::initBackend(MLLM_CPU);
        backend_ = Module::backends[MLLM_CPU];
        saved_list_idx = Module::listIdx;
        init_ = true;
    }
    bool ready() {
        return init_;
    }
    static map<string, string> layername_2_tensorname;

    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }

private:
    std::string name_num_to_X(const std::string &input_string) {
        std::regex pattern(R"(\.\d{1,3}\.)"); // Matches any number between 1 and 100 between two dots
        std::string replacement = ".X.";      // The string to replace the matched pattern with
        std::string output_string = std::regex_replace(input_string, pattern, replacement);
        return output_string;
    }
    std::string name_X_to_num(const std::string &input_string, int in_idx) {
        std::regex pattern(".X.");                                    // Matches any number between 1 and 100 between two dots
        std::string replacement = "." + std::to_string(in_idx) + "."; // The string to replace the matched pattern with
        std::string output_string = std::regex_replace(input_string, pattern, replacement);
        return output_string;
    }
    void reset_KVCache(string input_name) {
        vector<string> renameX_names;
        renameX_names.push_back(input_name);
        const vector<string> suffixs = {"-view", ".split-0", ".split-1", ".split-2"};
        for (auto suffix : suffixs) {
            if (input_name.rfind(suffix) == (input_name.size() - suffix.size())) {
                const auto r_name = input_name.substr(0, input_name.size() - suffix.size());
                renameX_names.push_back(r_name);
                break;
            }
        }
        for (const auto &x_name : renameX_names) {
            auto name = name_X_to_num(x_name, saved_list_idx);
            vector<int> shape = {Tensor::gph_[x_name].batch(), Tensor::gph_[x_name].head(), Tensor::gph_[x_name].sequence(), Tensor::gph_[x_name].dimension()};
            layername_2_tensorname[name] = name;
            Tensor::gph_[name] = Tensor(backend_);
            Tensor::gph_[name].initFrom(Tensor::gph_[x_name]);
            Tensor::gph_[name].setName(name);
            vector<Tensor *> new_chd_tensors = {};
            for (auto child : Tensor::gph_[x_name].childTensors()) {
                new_chd_tensors.push_back(&Tensor::gph_[name_X_to_num(child->name(), saved_list_idx)]);
            }
            Tensor::gph_[name].childTensors().clear();
            Tensor::gph_[name].childTensors() = new_chd_tensors;
            if (Tensor::gph_[x_name].aggregated() == true) {
                vector<shared_ptr<Tensor>> new_aggregated_tensors = {};
                for (const auto &aggregated_tensor : Tensor::gph_[x_name].aggregated_tensors()) {
                    new_aggregated_tensors.push_back(
                        std::shared_ptr<Tensor>(&Tensor::gph_[layername_2_tensorname[name_X_to_num(aggregated_tensor->name(), saved_list_idx)]], [](Tensor *) {}));
                }
                Tensor::gph_[name].addTensors(new_aggregated_tensors, Tensor::gph_[x_name].aggregated_dim());
            }
        }
    }

protected:
    bool INIT_OP() {
        if (op_ == nullptr) {
            op_ = backend_->opCreate(param_, name_);//_backend is a CPUBackend object. creates an operator (cpuembedding)
        }
        if (Module::doLoad) {
            op_->load(*Module::loader);// loads data from .mllm file into weights_ tensor in CPUEmbedding (op) object.
        }
        return Module::doLoad;
    }
    Tensor &_1I1O_OP(Tensor &input) { // in embedding generation operation, input tensor is the tensor having token ids for tokens in input sequence
        Module::runlistIdx = saved_list_idx;
        if (INIT_OP()) {
            return input;
        } else {
            string layer_next_name = "out-" + op_->name();
            if (Tensor::gph_.find(input.name()) != Tensor::gph_.end()) {
                Tensor::gph_[input.name()].status() = input.status();
            }
            switch (input.status()) {
            case TENSOR_STATIC_INIT: {
                if (Tensor::gph_.find(input.name()) == Tensor::gph_.end()) {//if input tensor name:'input' is not in the graph
                    Tensor::gph_[input.name()] = input;
                    Tensor::gph_[input.name()].setName(input.name());
                } else if (input.count() != Tensor::gph_[input.name()].count()) { // note: mostly, count() == batch() * head() * sequence() * dimension()
                    Tensor::gph_[input.name()] = input;
                    Tensor::gph_[input.name()].setName(input.name());
                }
                auto in_name = input.name();//'input' if we are talking about input tensor
                if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                    if (param_["type"] == KVCACHE) {
                        layername_2_tensorname[layer_next_name] = layer_next_name;
                        reset_KVCache(input.name());
                        in_name = name_X_to_num(in_name, saved_list_idx);
                    } else {
                        layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                }
                auto next_name = layername_2_tensorname[layer_next_name];
                if (Tensor::gph_.find(next_name) == Tensor::gph_.end()) {
                    Tensor::gph_[next_name] = Tensor(backend_);//new tensor for output(it is for the next layer)
                    Tensor::gph_[next_name].setName(next_name);
                }
                vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[in_name], [](Tensor *) {})};// In this context, &Tensor::gph_[in_name] is likely a raw pointer to a Tensor object that should not be deleted by the shared_ptr. The custom deleter [](Tensor *) {} effectively makes the shared_ptr a non-owning pointer, preventing it from performing any deletion.
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->reshape(shared_inputs, shared_outputs);
                op_->setUp(shared_inputs, shared_outputs);
                if (Tensor::gph_[next_name].aggregated() == false) {
                    assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                }
                break;
            }
            case TENSOR_STATIC_READY: {
                auto next_name = layername_2_tensorname[layer_next_name];
                assert(Tensor::gph_[input.name()].hostPtr<float>() != nullptr);
                vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[input.name()], [](Tensor *) {})};
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->execute(shared_inputs, shared_outputs);
                if (Tensor::gph_[next_name].aggregated() == false) {
                    assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                }
                break;
            }
            default: {
                break;
            }
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            Tensor::gph_[next_name].status() = Tensor::gph_[input.name()].status();
            // Tensor::gph_[next_name].saveNData<float>(layer_next_name);
            return Tensor::gph_[next_name];
        }
    }
    Tensor &_2I1O_OP(Tensor &input0, Tensor &input1) {
        Module::runlistIdx = saved_list_idx;
        if (INIT_OP()) {
            return input0;
        } else {
            string layer_next_name = "out-" + op_->name();
            if (Tensor::gph_.find(input0.name()) != Tensor::gph_.end()) {
                Tensor::gph_[input0.name()].status() = input0.status();
            }

            if (Tensor::gph_.find(input1.name()) != Tensor::gph_.end()) {
                Tensor::gph_[input1.name()].status() = input0.status();
            }
            if ((Tensor::gph_.find(input0.name()) != Tensor::gph_.end()) && Tensor::gph_.find(input1.name()) != Tensor::gph_.end()) {
                assert(input0.status() == input1.status());
            }
            switch (input0.status()) {
            case TENSOR_STATIC_INIT: {
                if (Tensor::gph_.find(input0.name()) == Tensor::gph_.end() || input0.count() != Tensor::gph_[input0.name()].count()) {
                    Tensor::gph_[input0.name()] = input0;
                    Tensor::gph_[input0.name()].setName(input0.name());
                }
                if (Tensor::gph_.find(input1.name()) == Tensor::gph_.end() || input1.count() != Tensor::gph_[input1.name()].count()) {
                    Tensor::gph_[input1.name()] = input1;
                    Tensor::gph_[input1.name()].setName(input1.name());
                }
                if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                    layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                }
                auto next_name = layername_2_tensorname[layer_next_name];
                if (Tensor::gph_.find(next_name) == Tensor::gph_.end()) {
                    Tensor::gph_[next_name] = Tensor(backend_);
                    Tensor::gph_[next_name].setName(next_name);
                }
                vector<shared_ptr<Tensor>> shared_inputs{
                    std::shared_ptr<Tensor>(&Tensor::gph_[input0.name()], [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&Tensor::gph_[input1.name()], [](Tensor *) {})};
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->reshape(shared_inputs, shared_outputs);
                op_->setUp(shared_inputs, shared_outputs);
                assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                break;
            }
            case TENSOR_STATIC_READY: {
                auto next_name = layername_2_tensorname[layer_next_name];
                vector<shared_ptr<Tensor>> shared_inputs{
                    std::shared_ptr<Tensor>(&Tensor::gph_[input0.name()], [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&Tensor::gph_[input1.name()], [](Tensor *) {})};
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->execute(shared_inputs, shared_outputs);
                assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                break;
            }
            default: {
                break;
            }
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            Tensor::gph_[next_name].status() = Tensor::gph_[input0.name()].status();
            // Tensor::gph_[input0.name()].saveNData<float>(input0.name());
            // Tensor::gph_[input1.name()].saveNData<float>(input1.name());
            // Tensor::gph_[next_name].saveNData<float>(layer_next_name);
            return Tensor::gph_[next_name];
        }
    }
    Tensor &_3I1O_OP(Tensor &input0, Tensor &input1, Tensor &input2) {
        Module::runlistIdx = saved_list_idx;
        if (INIT_OP()) {
            return input0;
        } else {
            string layer_next_name = "out-" + op_->name();
            if (Tensor::gph_.find(input0.name()) != Tensor::gph_.end()) {
                Tensor::gph_[input0.name()].status() = input0.status();
            }
            if (Tensor::gph_.find(input1.name()) != Tensor::gph_.end()) {
                Tensor::gph_[input1.name()].status() = input0.status();
            }
            if (Tensor::gph_.find(input2.name()) != Tensor::gph_.end()) {
                Tensor::gph_[input2.name()].status() = input0.status();
            }
            if ((Tensor::gph_.find(input0.name()) != Tensor::gph_.end()) && Tensor::gph_.find(input1.name()) != Tensor::gph_.end()) {
                assert(input0.status() == input1.status());
            }
            if ((Tensor::gph_.find(input0.name()) != Tensor::gph_.end()) && Tensor::gph_.find(input2.name()) != Tensor::gph_.end()) {
                assert(input0.status() == input2.status());
            }
            switch (input0.status()) {
            case TENSOR_STATIC_INIT: {
                if (Tensor::gph_.find(input0.name()) == Tensor::gph_.end() || input0.count() != Tensor::gph_[input0.name()].count()) {
                    Tensor::gph_[input0.name()] = input0;
                    Tensor::gph_[input0.name()].setName(input0.name());
                }
                if (Tensor::gph_.find(input1.name()) == Tensor::gph_.end() || input1.count() != Tensor::gph_[input1.name()].count()) {
                    Tensor::gph_[input1.name()] = input1;
                    Tensor::gph_[input1.name()].setName(input1.name());
                }
                if (Tensor::gph_.find(input2.name()) == Tensor::gph_.end() || input2.count() != Tensor::gph_[input0.name()].count()) {
                    Tensor::gph_[input2.name()] = input2;
                    Tensor::gph_[input2.name()].setName(input2.name());
                }
                if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                    layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                }
                auto next_name = layername_2_tensorname[layer_next_name];
                if (Tensor::gph_.find(next_name) == Tensor::gph_.end()) {
                    Tensor::gph_[next_name] = Tensor(backend_);
                    Tensor::gph_[next_name].setName(next_name);
                }
                vector<shared_ptr<Tensor>> shared_inputs{
                    std::shared_ptr<Tensor>(&Tensor::gph_[input0.name()], [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&Tensor::gph_[input1.name()], [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&Tensor::gph_[input2.name()], [](Tensor *) {})};
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->reshape(shared_inputs, shared_outputs);
                op_->setUp(shared_inputs, shared_outputs);
                assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                break;
            }
            case TENSOR_STATIC_READY: {
                auto next_name = layername_2_tensorname[layer_next_name];
                vector<shared_ptr<Tensor>> shared_inputs{
                    std::shared_ptr<Tensor>(&Tensor::gph_[input0.name()], [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&Tensor::gph_[input1.name()], [](Tensor *) {}),
                    std::shared_ptr<Tensor>(&Tensor::gph_[input2.name()], [](Tensor *) {})};
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->execute(shared_inputs, shared_outputs);
                assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                break;
            }
            default: {
                break;
            }
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            Tensor::gph_[next_name].status() = Tensor::gph_[input0.name()].status();
            // Tensor::gph_[next_name].saveNData<float>(layer_next_name);
            return Tensor::gph_[next_name];
        }
    }
    Tensor &_0I1O_OP() {
        Module::runlistIdx = saved_list_idx;
        if (INIT_OP()) {
            return Tensor::gph_["0"];
        } else {
            string layer_next_name = "param-" + op_->name();
            switch (Module::tensor_status) {
            case TENSOR_STATIC_INIT: {
                if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                    layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                }
                auto next_name = layername_2_tensorname[layer_next_name];
                if (Tensor::gph_.find(next_name) == Tensor::gph_.end()) {
                    Tensor::gph_[next_name] = Tensor(backend_);
                    Tensor::gph_[next_name].setName(next_name);
                }
                vector<shared_ptr<Tensor>> shared_inputs{};
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->reshape(shared_inputs, shared_outputs);
                op_->setUp(shared_inputs, shared_outputs);
                if (Tensor::gph_[next_name].aggregated() == false) {
                    assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                }
                break;
            }
            case TENSOR_STATIC_READY: {
                auto next_name = layername_2_tensorname[layer_next_name];
                vector<shared_ptr<Tensor>> shared_inputs{};
                vector<shared_ptr<Tensor>> shared_outputs{std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {})};
                op_->execute(shared_inputs, shared_outputs);
                if (Tensor::gph_[next_name].aggregated() == false) {
                    assert(Tensor::gph_[next_name].hostPtr<float>() != nullptr);
                }
                break;
            }
            default: {
                break;
            }
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            Tensor::gph_[next_name].status() = Module::tensor_status;
            // Tensor::gph_[next_name].saveNData<float>(layer_next_name);
            return Tensor::gph_[next_name];
        }
    }
    vector<Tensor> _1INO_OP(Tensor &input, int N) {
        Module::runlistIdx = saved_list_idx;
        if (INIT_OP()) {
            vector<Tensor> out;
            for (int i = 0; i < N; ++i) {
                out.push_back(input);
            }
            return out;
        } else {
            if (Tensor::gph_.find(input.name()) != Tensor::gph_.end()) {
                Tensor::gph_[input.name()].status() = input.status();
            }

            vector<string> layer_next_names = {};
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
            }
            switch (input.status()) {
            case TENSOR_STATIC_INIT: {
                if (Tensor::gph_.find(input.name()) == Tensor::gph_.end()) {
                    Tensor::gph_[input.name()] = input;
                    Tensor::gph_[input.name()].setName(input.name());
                } else if (input.count() != Tensor::gph_[input.name()].count()) {
                    Tensor::gph_[input.name()] = input;
                    Tensor::gph_[input.name()].setName(input.name());
                }
                vector<shared_ptr<Tensor>> shared_outputs = {};
                vector<string> next_names = {};
                for (const auto &layer_next_name : layer_next_names) {
                    if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                        layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                    auto next_name = layername_2_tensorname[layer_next_name];
                    if (Tensor::gph_.find(next_name) == Tensor::gph_.end()) {
                        Tensor::gph_[next_name] = Tensor(backend_);
                        Tensor::gph_[next_name].setName(next_name);
                    }
                    next_names.push_back(next_name);
                    shared_outputs.push_back(std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {}));
                }
                vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[input.name()], [](Tensor *) {})};
                op_->reshape(shared_inputs, shared_outputs);
                op_->setUp(shared_inputs, shared_outputs);
                break;
            }
            case TENSOR_STATIC_READY: {
                vector<shared_ptr<Tensor>> shared_outputs = {};
                vector<string> next_names = {};
                for (const auto &layer_next_name : layer_next_names) {
                    auto next_name = layername_2_tensorname[layer_next_name];
                    next_names.push_back(next_name);
                    shared_outputs.push_back(std::shared_ptr<Tensor>(&Tensor::gph_[next_name], [](Tensor *) {}));
                }
                if (Tensor::gph_[input.name()].aggregated() == false) {
                    assert(Tensor::gph_[input.name()].hostPtr<float>() != nullptr);
                }
                vector<shared_ptr<Tensor>> shared_inputs{std::shared_ptr<Tensor>(&Tensor::gph_[input.name()], [](Tensor *) {})};
                op_->execute(shared_inputs, shared_outputs);
                for (int i = 0; i < shared_outputs.size(); ++i) {
                    assert(Tensor::gph_[next_names[i]].hostPtr<float>() != nullptr);
                }
                break;
            }
            default: {
                break;
            }
            }
            vector<Tensor> output_result = {};
            for (const auto &layer_next_name : layer_next_names) {
                auto next_name = layername_2_tensorname[layer_next_name];
                Tensor::gph_[next_name].status() = Tensor::gph_[input.name()].status();
                // Tensor::gph_[next_name].saveNData<float>(layer_next_name);
                output_result.push_back(Tensor::gph_[next_name]);
            }
            return output_result;
        }
    }

    std::string name_;
    Op *op_ = nullptr;
    Backend *backend_{};
    OpParam param_;
    bool init_ = false;
    int saved_list_idx;
};

class Linear final : public Layer {
public:
    explicit Linear(int in_features, int out_features, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LINEAR);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class SiLU final : public Layer {
public:
    SiLU() = default;
    SiLU(std::string name) {
        init(std::move(name), OpType::SILU);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class ReLU final : public Layer {
public:
    ReLU() = default;
    ReLU(std::string name) {
        init(std::move(name), OpType::RELU);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class ReLUSquaredActivation final : public Layer {
public:
    ReLUSquaredActivation() = default;
    ReLUSquaredActivation(std::string name) {
        init(std::move(name), OpType::RELU2);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class GELU final : public Layer {
public:
    GELU() = default;
    GELU(std::string name) {
        init(std::move(name), OpType::OP_GELU);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class QuickGELU final : public Layer {
public:
    QuickGELU() = default;
    explicit QuickGELU(std::string name) {
        init(std::move(name), OpType::QUICKGLUE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

using ActFnConstructor = std::function<Layer(const std::string &)>;
inline std::map<std::string, ActFnConstructor> ACT_FN = {
    {"SiLU", [](const std::string &name) { return SiLU(name); }},
    {"ReLU", [](const std::string &name) { return ReLU(name); }},
    {"ReLU2", [](const std::string &name) { return ReLUSquaredActivation(name); }},
    {"GELU", [](const std::string &name) { return GELU(name); }},
    {"QuickGELU", [](const std::string &name) { return QuickGELU(name); }},
};

class Softmax final : public Layer {
public:
    explicit Softmax(Chl axis, std::string name) {
        param_["axis"] = axis;
        init(std::move(name), OpType::SOFTMAX);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Embedding final : public Layer {
public:
    explicit Embedding(int vocab_size, int hidden_size, std::string name) {
        param_["hidden_size"] = hidden_size;
        param_["vocab_size"] = vocab_size;
        init(std::move(name), OpType::EMBEDDING); // sets layer name as embedding and operation type as embedding
    }
    Tensor &operator()(Tensor &input) {// this method is called via embedding object in modeling_llama.hpp
        return _1I1O_OP(input);//input is input tokid tensor
    }
};

class Causalmask final : public Layer {
public:
    explicit Causalmask(std::string name) {
        init(std::move(name), OpType::CAUSALMASK);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class SlidingWindowMask final : public Layer {
public:
    explicit SlidingWindowMask(int window_size, std::string name) {
        param_["window_size"] = window_size;
        init(std::move(name), OpType::SLIDINGWINDOWMASK);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class RoPE final : public Layer {
public:
    explicit RoPE(int pose_type, std::string name) {
        param_["pose_type"] = pose_type;
        init(std::move(name), OpType::ROPE);
    }
    explicit RoPE(int pose_type, float rope_theta, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        init(std::move(name), OpType::ROPE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class KVCache final : public Layer {
public:
    explicit KVCache(int cache_max, std::string name) {
        param_["n_rep"] = 1;
        param_["cache_max"] = cache_max;
        init(std::move(name), OpType::KVCACHE);
    }
    explicit KVCache(int n_rep, int cache_max, std::string name) {
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        init(std::move(name), OpType::KVCACHE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class LayerNorm final : public Layer {
public:
    explicit LayerNorm(int norm_size, bool bias, float epsilon, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LAYERNORM);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class RMSNorm final : public Layer {
public:
    explicit RMSNorm(int norm_size, float epsilon, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        init(std::move(name), OpType::RMSNORM);
    }

    explicit RMSNorm(int norm_size, float epsilon, bool add_unit_offset, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        param_["add_unit_offset"] = (float)add_unit_offset;
        init(std::move(name), OpType::RMSNORM);
    }

    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Matmul final : public Layer {
public:
    explicit Matmul(bool transpose0, bool transpose1, std::string name) {
        param_["transpose0"] = transpose0;
        param_["transpose1"] = transpose1;
        init(std::move(name), OpType::MATMUL);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        return _2I1O_OP(input0, input1);
    }
};

class Split final : public Layer {
public:
    Split() = default;

    explicit Split(int split_num, Chl split_dim, int split_dim_size, std::string name) {
        param_["split_num"] = (float)split_num;
        param_["split_dim"] = (float)split_dim;
        param_["split_dim_size"] = (float)split_dim_size;
        init(std::move(name), OpType::SPLIT);
    }

    explicit Split(const std::vector<int> &each_dims, Chl split_dim, const std::string &name) {
        param_["split_num"] = (float)each_dims.size();
        param_["split_dim"] = (float)split_dim;
        // store each dims
        for (size_t i = 0; i < each_dims.size(); ++i) {
            param_["split_dim_size_" + std::to_string(i)] = (float)each_dims[i];
        }
        init(std::move(name), OpType::SPLIT);
    }

    vector<Tensor> operator()(Tensor &input) {
        return _1INO_OP(input, (int)param_["split_num"]);
    }
};

class Convolution2D final : public Layer {
public:
    explicit Convolution2D(int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, std::string name) {
        param_["in_channel"] = (float)in_channel;
        param_["out_channel"] = (float)out_channel;
        param_["kernal_h"] = (float)kernal[0];
        param_["kernal_w"] = (float)kernal[1];
        param_["stride_h"] = (float)stride[0];
        param_["stride_w"] = (float)stride[1];
        param_["padding"] = (float)padding;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::CONVOLUTION2D);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Convolution3D final : public Layer {
public:
    explicit Convolution3D(int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, std::string name) {
        param_["in_channel"] = (float)in_channel;
        param_["out_channel"] = (float)out_channel;
        param_["kernal_t"] = (float)kernal[0];
        param_["kernal_h"] = (float)kernal[1];
        param_["kernal_w"] = (float)kernal[2];
        param_["stride_t"] = (float)stride[0];
        param_["stride_h"] = (float)stride[1];
        param_["stride_w"] = (float)stride[2];
        param_["padding"] = (float)padding;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::CONVOLUTION3D);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Concat final : public Layer {
public:
    explicit Concat(Chl axis, std::string name) {
        param_["axis"] = (float)axis;
        init(std::move(name), OpType::CAT);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        return _2I1O_OP(input0, input1);
    }
};

class Parameter final : public Layer {
public:
    Parameter() = default;
    explicit Parameter(int batch, int seq, int head, int dim, std::string name) {
        param_["batch"] = batch;
        param_["seq"] = seq;
        param_["head"] = head;
        param_["dim"] = dim;
        init(std::move(name), OpType::PARAMETER);
    }
    Tensor &operator()() {
        return _0I1O_OP();
    }
};

} // namespace mllm

#endif // OPERATION_H
