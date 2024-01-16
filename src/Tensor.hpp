#ifndef MLLM_TENSOR_H
#define MLLM_TENSOR_H
#include <climits>
#include "Backend.hpp"
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include "Timing.hpp"
#include <fstream>
// #include <filesystem>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <assert.h>
#include <sys/stat.h>

const auto KMaxAxes = 32;

namespace mllm {
class Backend;

class Tensor {
public:
    Tensor() :
        host_ptr_(), capacity_(0), dtype_(MLLM_TYPE_F32) {
    }
    Tensor(Backend *bn) :
        backend_(bn), host_ptr_(), capacity_(0), dtype_(MLLM_TYPE_F32) {
    }
    ~Tensor() {
        if (host_ptr_ != nullptr && masterTensor() == nullptr && !aggregated_) {
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
        }
    }
    explicit Tensor(const int num, const int channels, const int height, const int width);
    explicit Tensor(const vector<int> &shape);

    bool reshape(const int batch, const int head, const int sequence, const int dimension);

    void alloc();
    void alloc(DataType dtype) {
        dtype_ = dtype;
        alloc();
    }

    void free() {
        if (aggregated_) { return; }
        if (host_ptr_ != nullptr && masterTensor() == nullptr) {
            backend_->free(host_ptr_);
            host_ptr_ = nullptr;
            allocated_ = 0;
        }
    }

    void update();

    size_t size() const {
        return capacity_ * dtypeSize();
    }

    inline int batch() const {
        if (ctype_ == SBHD) {
            return legacyShape(1);
        } else {
            return legacyShape(0);
        }
    }
    inline int head() const {
        switch (ctype_) {
        case BSHD:
            return legacyShape(2);
        case BHDS:
            return legacyShape(1);
        case SBHD:
            return legacyShape(2);
        default:
            return -1;
        }
    }
    inline int sequence() const {
        switch (ctype_) {
        case BSHD:
            return legacyShape(1);
        case BHDS:
            return legacyShape(3);
        case SBHD:
            return legacyShape(0);
        default:
            return -1;
        }
    }
    inline int dimension() const {
        switch (ctype_) {
        case BSHD:
            return legacyShape(3);
        case BHDS:
            return legacyShape(2);
        case SBHD:
            return legacyShape(3);
        default:
            return -1;
        }
    }

    inline int count() const {
        return count_;
    }
    inline int numAxes() const {
        return shape_.size();
    }
    inline string ShapeString() const {
        std::ostringstream stream;
        for (int i : shape_) {
            stream << i << " ";
        }
        stream << "(" << count_ << ")";
        return stream.str();
    }
    inline int canonicalAxisIndex(int axis_index) const {
        if (axis_index < 0) {
            return axis_index + numAxes();
        }
        return axis_index;
    }
    inline int legacyShape(int index) const {
        if (index >= numAxes() || index < -numAxes()) {
            return 1;
        }
        return shape(index);
    }
    inline int offset(const int b, const int h = 0, const int s = 0,
                      const int d = 0) const {
        // batch, head, sequence, dimension
        if (shape_offset_.size() == 4 & shape_master_.size() == 4) {
            const int base_batch_ = shape_master_[0];
            const int base_head_ = shape_master_[1];
            const int base_sequence_ = shape_master_[2];
            const int base_dimension_ = shape_master_[3];
            const int b_ = (b + shape_offset_[0]) % base_batch_;
            const int h_ = (h + shape_offset_[1]) % base_head_;
            const int s_ = (s + shape_offset_[2]) % base_sequence_;
            const int d_ = (d + shape_offset_[3]) % base_dimension_;
            switch (ctype_) {
            case BSHD:
                return ((b_ * base_sequence_ + s_) * base_head_ + h_) * base_dimension_ + d_;
            case BHDS:
                return ((b_ * base_head_ + h_) * base_dimension_ + d_) * base_sequence_ + s_;
            case SBHD:
                return ((s_ * base_batch_ + b_) * base_head_ + h_) * base_dimension_ + d_;
            default:
                break;
            }
        } else {
            switch (ctype_) {
            case BSHD:
                return ((b * shape_[1] + s) * shape_[2] + h) * shape_[3] + d;
            case BHDS:
                return ((b * shape_[1] + h) * shape_[2] + d) * shape_[3] + s;
            case SBHD:
                return ((s * shape_[1] + b) * shape_[2] + h) * shape_[3] + d;
            default:
                break;
            }
        }
        return -1;
    }

    inline int offset(const vector<int> &indices) const {
        if (shape_offset_.size() == 4 & shape_master_.size() == 4) {
            return offset(indices[0], indices[1], indices[2], indices[3]);
        } else {
            int offset = 0;
            for (int i = 0; i < numAxes(); ++i) {
                offset *= shape(i);
                if (indices.size() > i) {
                    offset += indices[i];
                }
            }
            return offset;
        }
    }

    template <typename Dtype>
    Dtype *hostPtr() const {
        return (Dtype *)host_ptr_;
    }

    template <typename Dtype>
    Dtype dataAt(const int batch, const int head, const int sequence, const int dimension) const {
        if (!aggregated_) {
            return ((Dtype *)host_ptr_)[offset(batch, head, sequence, dimension)];
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->dataAt<Dtype>(b, h, s, d);
        }
    }

    template <typename Dtype>
    Dtype dataAt(const vector<int> &index) const {
        return dataAt<Dtype>(index[0], index[1], index[2], index[3]);
    }

    template <typename Dtype>
    Dtype *ptrAt(const int batch, const int head, const int sequence, const int dimension) {
        if (!aggregated_) {
            return ((Dtype *)host_ptr_ + offset(batch, head, sequence, dimension));
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->ptrAt<Dtype>(b, h, s, d);
        }
    }

    template <typename Dtype>
    Dtype *ptrAt(const vector<int> &index) const {
        return ptrAt<Dtype>(index[0], index[1], index[2], index[3]);
    }

    template <typename Dtype>
    void setDataAt(const int batch, const int head, const int sequence, const int dimension, Dtype value) {
        if (!aggregated_) {
            Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
            typed_ptr[offset(batch, head, sequence, dimension)] = value;
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            aggregated_tensors_[tensor_id]->setDataAt<Dtype>(b, h, s, d, value);
        }
    }

    template <typename Dtype>
    void setDataAt(const vector<int> &index, Dtype value) {
        setDataAt(index[0], index[1], index[2], index[3], value);
    }
    DataType dtypeAt(const int batch, const int head, const int sequence, const int dimension) const {
        if (!aggregated_) {
            return dtype_;
        } else {
            int b = batch;
            int h = head;
            int s = sequence;
            int d = dimension;
            int tensor_id = checkDim(b, h, s, d);
            return aggregated_tensors_[tensor_id]->dtype_;
        }
    }

    Backend *backend() const {
        return backend_;
    }
    void setBackend(Backend *bn) {
        backend_ = bn;
    };

    DataType dtype() const {
        return dtype_;
    }
    void setDtype(DataType dtype) {
        dtype_ = dtype;
    }

    inline const vector<int> &shape() const {
        return shape_;
    }

    ChlType ctype() const {
        return ctype_;
    }
    void setCtype(ChlType type) {
        ctype_ = type;
    }
    size_t cntSize() {
        return DataTypeSize(dtype_, count_);
    }
    int dtypeSize() const {
        return DataTypeSize(dtype_, 1);
    }
    int dtypeSize(int size) {
        return DataTypeSize(dtype_, size);
    }
    void setName(string name) {
        name_ = name;
    }
    string name() const {
        return name_;
    }
    int allocted() const {
        return allocated_;
    }

    /**
     * @brief Copy from a source Tensor.
     * @param source the Tensor to copy from
     * @param copy_diff if false, copy the data; if true, copy the diff
     * @param reshape if false, require this Tensor to be pre-shaped to the shape
     *        of other (and die otherwise); if true, reshape this Tensor to other's
     *        shape if necessary
     */
    void copyFrom(const Tensor &source, bool copy_diff = false, bool reshape = false) {
        assert(masterTensor() == nullptr);
        assert(source.dtype() == dtype());
        assert(source.count() == count());
        // copy
        memcpy(host_ptr_, source.host_ptr_, cntSize());
    }
    void copyFrom(const shared_ptr<Tensor> &source, bool reshape = false) {
        assert(masterTensor() == nullptr);
        assert(source->dtype() == dtype());
        assert(source->count() == count());
        // copy
        memcpy(host_ptr_, source->host_ptr_, cntSize());
    }

    /************************************Deep Copy   MasterTenser  *******************/

    /**
     * \brief this Tensor is a DEEP COPY of source
     * \param source
     * \param shape_offset
     */

    void deepCopyFrom(Tensor *source, bool copyshape = true, const vector<int> &shape_offset = {}, int head_rep = 1) {
        if (!shape_offset.empty()) {
            copyshape = false;
        }
        setMasterTensor(source);
        if (ctype_ != BCTHW && ctype_ != BTHWC && ctype_ != master_tensor_->ctype() && undiffusion_ == false) {
            if (transed_) { // child tensor have been transed(BSHD->BHDS);
                auto b = master_tensor_->batch();
                auto h = master_tensor_->head();
                auto d = master_tensor_->dimension();
                auto s = master_tensor_->sequence();
                master_tensor_->ctype_ = ctype_;
                master_tensor_->reshape(b, h, s, d);
            } else {
                auto b = batch();
                auto h = head();
                auto d = dimension();
                auto s = sequence();
                ctype_ = master_tensor_->ctype_;
                reshape(b, h, s, d);
            }
        }
        host_ptr_ = source->hostPtr<void>();
        capacity_ = source->capacity_;
        count_ = source->count_;
        if (copyshape) {
            shape_ = source->shape_;
        }
        allocated_ = source->allocated_;
        dtype_ = source->dtype_;

        if (!shape_offset.empty()) {
            shape_offset_ = shape_offset;
            shape_master_ = {source->batch(), source->head(), source->sequence(), source->dimension()};
            if (source->head() != head()) { // TODO: need to check
                if (head() == 1 && head_rep == 1) {
                    shape_master_ = {source->batch(), head(), source->sequence(), source->dimension() * source->head() / head()};
                } else if (head() == 1 && head_rep > 1) {
                    shape_master_ = {source->batch(), head(), source->sequence(), source->dimension() * source->head() / head_rep};
                }
            }
        }

        for (auto &child_tensor : child_tensors_) {
            if (!shape_offset.empty()) {
                child_tensor->deepCopyFrom(source, false, shape_offset, head_rep);
            } else {
                child_tensor->deepCopyFrom(source, false, {}, head_rep);
            }
            child_tensors_.erase(std::remove(child_tensors_.begin(), child_tensors_.end(), child_tensor), child_tensors_.end());
        }
        source->addChildTensor(this);
    }

    void deepCopyFrom(Tensor &source, bool copyshape = true, const vector<int> &shape_offset = {}, int head_rep = 1) {
        deepCopyFrom(&source, copyshape, shape_offset, head_rep);
    }

    vector<int> shape_offset() const {
        return shape_offset_;
    }
    vector<int> shape_master() const {
        return shape_master_;
    }

    Tensor *masterTensor() const {
        return master_tensor_;
    }
    void setMasterTensor(Tensor *master_tensor) {
        master_tensor_ = master_tensor;
    }

    vector<Tensor *> childTensors() {
        return child_tensors_;
    }
    void addChildTensor(Tensor *child) {
        child_tensors_.push_back(child);
    }

    void transShape(Chl dim_a = SEQUENCE, Chl dim_b = DIMENSION, bool undiffusion = false) {
        if (dim_a == SEQUENCE && dim_b == DIMENSION && ctype() == BSHD) {
            auto b = batch();
            auto h = head();
            auto d = dimension();
            auto s = sequence();
            ctype_ = BHDS;
            reshape(b, h, s, d);
            transed_ = true;
            undiffusion_ = undiffusion;
        } else if (THW == dim_a && dim_b == CHANNLE && ctype() == BCTHW) {
            auto b = batch();
            auto c = channel();
            auto t = time();
            auto h = height();
            auto w = width();
            ctype_ = BTHWC;
            reshape(b, c, t, h, w);
            transed_ = true;
            undiffusion_ = undiffusion;
        } else if (dim_a == BATCH && dim_b == SEQUENCE && ctype() == BSHD) {
            auto b = batch();
            auto h = head();
            auto d = dimension();
            auto s = sequence();
            ctype_ = SBHD;
            reshape(b, h, s, d);
            transed_ = true;
            undiffusion_ = undiffusion;
        }
    }

    /************************************Aggregated  Tensers  *******************/

    void addTensors(vector<shared_ptr<Tensor>> ts, Chl dim) {
        aggregated_ = true;
        aggregated_dim_ = dim;
        aggregated_dims_ = {};
        switch (dim) {
        case HEAD: {
            auto sum = 0;
            for (auto &t : ts) {
                assert(t->batch() == batch());
                assert(t->sequence() == sequence());
                assert(t->dimension() == dimension());
                sum += t->head();
                aggregated_dims_.push_back(sum);
            }
            assert(sum == head());
            break;
        }
        case SEQUENCE: {
            auto sum = 0;
            for (auto &t : ts) {
                assert(t->batch() == batch());
                assert(t->head() == head());
                assert(t->dimension() == dimension());
                sum += t->sequence();
                aggregated_dims_.push_back(sum);
            }
            assert(sum == sequence());
            break;
        }
        case DIMENSION: {
            auto sum = 0;
            for (auto &t : ts) {
                assert(t->batch() == batch());
                assert(t->head() == head());
                assert(t->sequence() == sequence());
                sum += t->dimension();
                aggregated_dims_.push_back(sum);
            }
            assert(sum == dimension());
            break;
        }
        default:
            break;
        }
        aggregated_tensors_ = ts;
    }

    /************************************B, C, T, H, W  *******************/

    bool reshape(const int batch, const int channel, const int time, const int height, const int width);
    int channel() const {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        switch (ctype_) {
        case BCTHW:
            return legacyShape(1);
        case BTHWC:
            return legacyShape(4);
        default: return -1;
        }
    }
    int time() const {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        switch (ctype_) {
        case BCTHW:
            return legacyShape(2);
        case BTHWC:
            return legacyShape(1);
        default: return -1;
        }
    }
    int height() const {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        switch (ctype_) {
        case BCTHW:
            return legacyShape(3);
        case BTHWC:
            return legacyShape(2);
        default: return -1;
        }
    }
    int width() const {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        switch (ctype_) {
        case BCTHW:
            return legacyShape(4);
        case BTHWC:
            return legacyShape(3);
        default: return -1;
        }
    }
    int offset(const int b, const int c, const int t, const int h, const int w) const {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        switch (ctype_) {
        case BCTHW:
            return (((b * channel() + c) * time() + t) * height() + h) * width() + w;
        case BTHWC:
            return (((b * time() + t) * height() + h) * width() + w) * channel() + c;
        default: return -1;
        }
    }
    template <typename Dtype>
    Dtype dataAt(const int batch, const int channel, const int time, const int height, const int width) const {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        return ((Dtype *)host_ptr_)[offset(batch, channel, time, height, width)];
    }
    template <typename Dtype>
    Dtype *ptrAt(const int batch, const int channel, const int time, const int height, const int width) {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        return ((Dtype *)host_ptr_ + offset(batch, channel, time, height, width));
    }
    template <typename Dtype>
    void setDataAt(const int batch, const int channel, const int time, const int height, const int width, Dtype value) {
        assert(ctype_ == BCTHW || ctype_ == BTHWC);
        Dtype *typed_ptr = static_cast<Dtype *>(host_ptr_);
        typed_ptr[offset(batch, channel, time, height, width)] = value;
    }

public:
    /*TEST*/

    /************************************ TEST & DEBUG  *******************/

    template <typename Dtype>
    void checkData() {
        if (ctype() == BTHWC || ctype() == BCTHW || dtype() != MLLM_TYPE_F32) {
            return;
        }
        // n c h w
        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        bool ck = false;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float value = dataAt<Dtype>(n, c, h, w);
                        if (std::isnan(value) || std::isnan(-value)) {
                            // std::cout<<"["<<n<<","<<c<<","<<h<<","<<w<<"] ";//<<std::flush;
                            ck = true;
                        }
                    }
                }
            }
        }
        if (ck) {
            std::cout << "\n[ERROR]:" << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] has Nan" << std::endl;
            // printData<Dtype>();
            assert(ck == false);
        }
    }

    void printShape() {
        std::cout << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "]" << std::endl;
    }

    template <typename Dtype>
    void printData() {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "]" << std::endl;
        // n c h w
        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        if (N == 1 && C == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < W; ++c) {
                    std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(0, 0, h, c) << " ";
                }
                std::cout << std::endl;
                std::cout << "---------" << std::endl;
            }
        } else if (N == 1 && W == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < C; ++c) {
                    std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(0, c, h, 0) << " ";
                }
                std::cout << std::endl;
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(n, c, h, w) << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }

    template <typename Dtype>
    void saveData(string ex = "") {
        if (ctype() == BTHWC || ctype() == BCTHW) {
            save5Data<Dtype>(ex);
            return;
        }
        // std::filesystem::create_directory("save_out");
        string directory = "save_out";
        struct stat info;

        if (stat(directory.c_str(), &info) != 0) {
            // if the directory does not exist, create it
#ifdef _WIN32
            _mkdir(directory.c_str());
#else
            mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
#endif
        } else if (!(info.st_mode & S_IFDIR)) {
            // if the path exists but it is not a directory, also create it
#ifdef _WIN32
            _mkdir(directory.c_str());
#else
            mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
#endif
        }
        std::ofstream outFile(directory + "/" + name() + ex + ".log");

        outFile << "----------------------------------------" << std::endl;
        outFile << name() << ": shape:[" << batch() << " " << head() << " " << sequence() << " " << dimension() << "] " << dtype() << " " << ctype() << std::endl;

        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        if (N == 1 && C == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < W; ++c) {
                    outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, 0, h, c) << " ";
                }
                outFile << std::endl;
                outFile << "---------" << std::endl;
            }
        } else if (N == 1 && W == 1) {
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < C; ++c) {
                    outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(0, c, h, 0) << " ";
                }
                outFile << std::endl;
            }
        } else {
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            outFile << std::fixed << std::setprecision(6) << dataAt<Dtype>(n, c, h, w) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
        }

        outFile.close();
    }

    /************************************B, C, T, H, W  *******************/
    template <typename Dtype>
    void print5Data() {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << name() << ": shape:[" << batch() << " " << channel() << " " << time() << " " << height() << " " << width() << "]" << std::endl;
        int N = batch();
        int C = channel();
        int T = time();
        int H = height();
        int W = height();
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < T; ++t) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            std::cout << std::fixed << std::setprecision(7) << dataAt<Dtype>(n, c, t, h, w) << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }

    template <typename Dtype>
    void save5Data(string ex = "") {
        // std::filesystem::create_directory("save_out");
        string directory = "save_out";
        struct stat info;

        if (stat(directory.c_str(), &info) != 0) {
            // if the directory does not exist, create it
#ifdef _WIN32
            _mkdir(directory.c_str());
#else
            mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
#endif
        } else if (!(info.st_mode & S_IFDIR)) {
            // if the path exists but it is not a directory, also create it
#ifdef _WIN32
            _mkdir(directory.c_str());
#else
            mkdir(directory.c_str(), 0777); // notice that 0777 is different than usual
#endif
        }
        std::ofstream outFile(directory + "/" + name() + ex + ".log");
        outFile << "----------------------------------------" << std::endl;
        outFile << name() << ": shape:[" << batch() << " " << channel() << " " << time() << " " << height() << " " << width() << "]" << std::endl;
        int N = batch();
        int C = channel();
        int T = time();
        int H = height();
        int W = height();
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int t = 0; t < T; ++t) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            outFile << std::fixed << std::setprecision(7) << dataAt<Dtype>(n, c, t, h, w) << " ";
                        }
                        outFile << std::endl;
                    }
                    outFile << std::endl;
                }
                outFile << std::endl;
            }
        }
    }

    template <typename Dtype>
    void printMem() {
        for (int i = 0; i < count_; ++i) {
            auto *typed_ptr = static_cast<Dtype *>(host_ptr_);
            std::cout << std::fixed << std::setprecision(7) << typed_ptr[i] << " ";
        }
    }

    template <typename Dtype>
    void printAVG() {
        float sum = 0;
        // n c h w
        int N = batch();
        int C = head();
        int H = sequence();
        int W = dimension();
        bool ck = false;
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float value = dataAt<Dtype>(n, c, h, w);
                        sum += value;
                    }
                }
            }
        }
        std::cout << name() << " " << sum / count() << std::endl;
    }

    shared_ptr<Tensor> view(int batch, int head, int sequence, int dimension) {
        auto t = std::make_shared<Tensor>();
        t->setBackend(backend_);
        t->setDtype(dtype_);
        t->reshape(batch, head, sequence, dimension);
        t->host_ptr_ = host_ptr_;
        return t;
    }

    template <class Dtype>
    void fullData(Dtype value) {
        for (int n = 0; n < batch(); ++n) {
            for (int c = 0; c < head(); ++c) {
                for (int h = 0; h < sequence(); ++h) {
                    for (int w = 0; w < dimension(); ++w) {
                        setDataAt<Dtype>(n, c, h, w, value);
                    }
                }
            }
        }
    }

    void fullDataTest() {
        for (int n = 0; n < batch(); ++n) {
            for (int c = 0; c < head(); ++c) {
                for (int h = 0; h < sequence(); ++h) {
                    for (int w = 0; w < dimension(); ++w) {
                        setDataAt<float>(n, c, h, w, offset(n, c, h, w));
                    }
                }
            }
        }
    }
    void fullDataTest2() {
        for (int i = 0; i < count_; ++i) {
            float *typed_ptr = static_cast<float *>(host_ptr_);
            typed_ptr[i] = i;
        }
    }

private:
    bool reshape(const vector<int> &shape) {
        assert(shape.size() <= KMaxAxes);
        count_ = 1;
        shape_.resize(shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            assert(shape[i] >= 0);
            if (count_ != 0) {
                assert(shape[i] <= INT_MAX / count_);
            }
            count_ *= shape[i]; // 记录数据大小
            shape_[i] = shape[i];
        }
        if (count_ > capacity_) { // capactity不小于count
            capacity_ = count_;
            return true;
        }
        return false;
    }
    inline int shape(int index) const {
        return shape_[canonicalAxisIndex(index)];
    }

    int checkDim(int &b, int &h, int &s, int &d) const {
        if (!aggregated_) {
            return -1;
        }
        int tensor_id = -1;
        switch (aggregated_dim_) {
        case HEAD: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (h < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            h = h - aggregated_dims_[tensor_id - 1];
            break;
        }
        case SEQUENCE: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (s < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            s = s - aggregated_dims_[tensor_id - 1];
            break;
        }
        case DIMENSION: {
            for (int a = 0; a < aggregated_dims_.size(); ++a) {
                if (d < aggregated_dims_[a]) {
                    tensor_id = a;
                    break;
                }
            }
            d = d - aggregated_dims_[tensor_id - 1];
            break;
        }
        case D_HD: {
            int dim_size = aggregated_tensors_[0]->dimension();
            int aggregated_size = aggregated_tensors_.size();
            auto h_ = d / (dim_size * aggregated_size);
            auto d_m = d % (dim_size * aggregated_size);
            tensor_id = d_m / dim_size;
            d = d_m % dim_size;
            h = h_;
            break;
        }
        case HD: {
            auto orin_d = d;
            int dim_size = aggregated_tensors_[0]->dimension();
            int head_size = aggregated_tensors_[0]->head();
            tensor_id = orin_d / (dim_size * head_size);
            h = (orin_d - tensor_id * (dim_size * head_size)) / dim_size;
            d = (orin_d - tensor_id * (dim_size * head_size)) % dim_size;
            break;
        }
        default:
            break;
        }
        return tensor_id;
    }

private:
    string name_;
    DataType dtype_;
    ChlType ctype_ = BSHD;
    Backend *backend_;
    void *host_ptr_;
    void *device_ptr_;
    vector<int> shape_;
    int capacity_;
    int count_;
    int allocated_ = 0;

    // shadow tensor if;
    vector<int> shape_offset_;
    vector<int> shape_master_;
    Tensor *master_tensor_ = nullptr;
    vector<Tensor *> child_tensors_;
    bool transed_ = false;
    bool undiffusion_ = false;

    // aggregated
    bool aggregated_ = false;
    vector<shared_ptr<Tensor>> aggregated_tensors_;
    Chl aggregated_dim_;
    vector<int> aggregated_dims_ ;

};
} // namespace mllm
#endif // MLLM_TENSOR_H