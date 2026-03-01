#ifndef __FAST_CUSTOM_VECTOR__INCLUDED__
#define __FAST_CUSTOM_VECTOR__INCLUDED__

#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

#ifdef Debug
#include <cassert>
#include <iostream>
#endif

namespace UserVector {

template <class T>
class Vector {
   private:
    size_t size_;
    size_t capacity_;
    double load_factor_;
    std::unique_ptr<T[]> memory_;

   public:
    // Ctors

    Vector() : size_(0), load_factor_(1.0), capacity_(0), memory_(nullptr) {}

    Vector(Vector&& other) noexcept : Vector() {
        if (this != &other) {
            memory_ = std::move(other.memory_);
            size_ = other.size_;
            capacity_ = other.capacity_;
            load_factor_ = other.load_factor_;
        }
    }

    Vector(size_t size, T default_value) : Vector() {
        Resize(size, default_value);
    }

    Vector(size_t size) : Vector() { Resize(size); }

    Vector(const Vector& other) : Vector() {
        Resize(other.capacity_);
        Fill(other);
        size_ = other.size_;
        capacity_ = other.capacity_;
        load_factor_ = other.load_factor_;
    }

    ~Vector() noexcept { Releasing_objects(memory_); }

    // Standard vector operators

    T& operator[](size_t pos) {
        if (pos >= size_) {
            throw std::out_of_range("Position is out of bounds.");
        }

        return *(memory_.get() + pos);
    }

    Vector operator=(Vector& other) {
        Reserve(other.capacity_);
        Fill(other);
        size_ = other.size_;
        capacity_ = other.capacity_;
        load_factor_ = other.load_factor_;
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            Releasing_objects(memory_);
            this->memory_ = std::move(other.memory_);
            size_ = other.size_;
            capacity_ = other.capacity_;
            load_factor_ = other.load_factor_;

            // //cleaning for other.
            other.size_ = 0;
            other.capacity_ = 0;
            other.load_factor_ = 1.0;
            other.memory_ = nullptr;
        }
        return *this;
    }

    // Standart vector functions

    void Insert(size_t pos, T x) {
        if (pos > size_ + 1) {
            // TODO: throw wet towel!
            throw std::out_of_range("Position is out of bounds.");
        }
        size_++;
        if (size_ >= capacity_) {
            Reserve(Count_capacity(size_));
        }
        for (size_t pos_ = size_ - 1; pos_ > pos; --pos_) {
            std::swap((*this)[pos_], (*this)[pos_ - 1]);
        }
        (*this)[pos] = x;
    }

    void Erase(size_t pos) {
        if (Empty()) {
            // TODO: throw wet towel!
            throw std::runtime_error("Can't erase, vector already empty.");
        }
        if (pos > size_) {
            // TODO: throw wet towel!
            throw std::runtime_error("Can't erase, position unexist.");
        }
        for (size_t pos_ = pos + 1; pos_ < size_; ++pos_) {
            std::swap((*this)[pos_], (*this)[pos_ - 1]);
        }
        (*this)[size_ - 1] = T();
        size_--;
    }

    void Emplace(size_t pos, T&& x) {
        if (pos > size_ + 1) {
            // TODO: throw wet towel!
            throw std::out_of_range("Position is out of bounds.");
        }
        size_++;
        if (size_ >= capacity_) {
            Reserve(Count_capacity(size_));
        }
        for (size_t pos_ = size_ - 1; pos_ > pos; --pos_) {
            std::swap((*this)[pos_], (*this)[pos_ - 1]);
        }
        (*this)[pos] = std::move(x);
    }

    void Push_back(T x) { Insert(size_, x); }

    void Push_front(T x) { Insert(0, x); }

    void Emplace_back(T&& x) { Emplace(size_, std::move(x)); }

    void Clear() { size_ = 0; }

    void Pop_front() { Erase(0); }

    void Pop_back() { Erase(size_ - 1); }

    T Front() {
        if (Empty()) {
            throw std::runtime_error("Vector is empty.");
        }
        return (*this)[0];
    }

    T Back() {
        if (Empty()) {
            throw std::runtime_error("Vector is empty.");
        }
        return (*this)[size_ - 1];
    }

    // Size related functions

    size_t Capacity() { return capacity_; }

    void Reserve(size_t capacity) {
        // There is no need to resize current block just because user wants.
        // Only if memory size is insuffitient
        if (capacity < capacity_) {
            return;
        }
        // Allocating new block of memory
        std::unique_ptr<T[]> memory1_ = std::make_unique<T[]>(capacity);
        swap(memory1_, memory_);  // Making new allocated block main
        Fill(memory1_);           // fill old data to new block of memory.
        capacity_ = capacity;
        Releasing_objects(memory1_);
        // we assing capacity later, because we need to remember capacity of old
        // block.
    }

    void Assign(size_t size, T& value) {
        if (size > capacity_) {
            Reserve(Count_capacity(size));
            size_ = size;
        }
        if (size > size_) {
            size_ = size;
        }
        for (size_t pos = 0; pos < size_; pos++) {
            (*this)[pos] = value;
        }
    }

    void Resize(size_t size, T default_value) {
        if (size >= capacity_) {
            Reserve(Count_capacity(size));
            size_ = size;
        }
        size_ = size;
        Assign(size_, default_value);
    }

    void Resize(size_t size) {
        if (size >= capacity_) {
            Reserve(Count_capacity(size));
        }
        size_ = size;
    }

    size_t Size() { return size_; }

    bool Empty() { return (size_ == 0); }

    void Change_load_factor(double x) {
        if (x < 0.0) {
            return;
        }
        load_factor_ = x;
    }

   private:
    void Releasing_objects(std::unique_ptr<T[]>& memory) {
        if (memory.get() == nullptr) {
            return;
        }
        if (!std::is_pod<T>::value) {  // Making optimization for pod types
            for (size_t pos = 0; pos < capacity_; pos++) {
                (memory.get() + pos)->~T();
            }
        }
        memory.release();
    }

    void Fill(const Vector& other) {
        if (other.memory_.get() == nullptr) {
            return;
        }
        for (size_t i = 0; i < other.capacity_; i++) {
            *(memory_.get() + i) = std::move(*(other.memory_.get() + i));
        }
    }

    void Fill(std::unique_ptr<T[]>& memory1_) {
        if (memory1_.get() == nullptr) {
            return;
        }
        for (size_t i = 0; i < capacity_; i++) {
            *(memory_.get() + i) = std::move(*(memory1_.get() + i));
        }
    }

    size_t Count_capacity(size_t size) {
        return static_cast<int>(size * (1 + load_factor_));
    }
};
}  // namespace UserVector

#endif