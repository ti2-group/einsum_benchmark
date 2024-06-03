#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cstdio>
#include <cstring>
#include <future>
#include <bitset>

template <typename Key, typename T>
class ConsistentOrderedMap {
private:
  std::vector<std::pair<Key, T>> data;
  std::unordered_map<Key, uint64_t> index_map;

public:
  ConsistentOrderedMap() = default;

  void insert(const std::pair<Key, T>& pair) {
    if (index_map.find(pair.first) == index_map.end()) {
      data.push_back(pair);
      index_map[pair.first] = data.size() - 1;
    }
  }

  T& operator[](const Key& key) {
    if (index_map.find(key) == index_map.end()) {
      data.push_back(std::make_pair(key, T()));
      index_map[key] = data.size() - 1;
    }
    return data[index_map[key]].second;
  }

  const T& operator[](const Key& key) const { return data[index_map.at(key)].second; }

  void erase(const Key& key) {
    auto it = index_map.find(key);
    if (it != index_map.end()) {
      uint64_t index = it->second;
      index_map.erase(it);
      if (index != data.size() - 1) {
        std::swap(data[index], data[data.size() - 1]);
        index_map[data[index].first] = index;
      }
      data.pop_back();
    }
  }

  typename std::vector<std::pair<Key, T>>::iterator begin() { return data.begin(); }

  typename std::vector<std::pair<Key, T>>::iterator end() { return data.end(); }

  typename std::vector<std::pair<Key, T>>::const_iterator begin() const { return data.begin(); }

  typename std::vector<std::pair<Key, T>>::const_iterator end() const { return data.end(); }

  typename std::vector<std::pair<Key, T>>::iterator find(const Key& key) {
    auto it = index_map.find(key);
    if (it != index_map.end()) {
      return data.begin() + it->second;
    }
    return data.end();
  }

  typename std::vector<std::pair<Key, T>>::const_iterator find(const Key& key) const {
    auto it = index_map.find(key);
    if (it != index_map.end()) {
      return data.begin() + it->second;
    }
    return data.end();
  }

  uint64_t size() const { return data.size(); }
};

template <typename Key>
class ConsistentOrderedSet {
private:
  std::vector<Key> data;
  std::unordered_map<Key, uint64_t> index_map;

public:
  ConsistentOrderedSet() = default;

  bool insert(const Key& key) {
    if (index_map.find(key) == index_map.end()) {
      data.push_back(key);
      index_map[key] = data.size() - 1;
      return true;
    }
    return false;
  }

  bool erase(const Key& key) {
    auto it = index_map.find(key);
    if (it != index_map.end()) {
      uint64_t index = it->second;
      index_map.erase(it);
      if (index != data.size() - 1) {
        Key lastKey = data.back();
        data[index] = lastKey;
        index_map[lastKey] = index;
      }
      data.pop_back();
      return true;
    }
    return false;
  }

  typename std::vector<Key>::iterator begin() { return data.begin(); }

  typename std::vector<Key>::iterator end() { return data.end(); }

  typename std::vector<Key>::const_iterator begin() const { return data.begin(); }

  typename std::vector<Key>::const_iterator end() const { return data.end(); }

  typename std::vector<Key>::iterator find(const Key& key) {
    auto it = index_map.find(key);
    if (it != index_map.end()) {
      return data.begin() + it->second;
    }
    return data.end();
  }

  typename std::vector<Key>::const_iterator find(const Key& key) const {
    auto it = index_map.find(key);
    if (it != index_map.end()) {
      return data.begin() + it->second;
    }
    return data.end();
  }

  void clear() {
    data.clear();
    index_map.clear();
  }

  bool contains(const Key& key) const { return index_map.find(key) != index_map.end(); }

  uint64_t size() const { return data.size(); }
};

// random number generator
inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

inline uint64_t next_rand(uint64_t& s0, uint64_t& s1) {
  s1 = s1 ^ s0;
  s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16);
  s1 = rotl(s1, 37);
  return s0 + s1;
}

inline double next_double_rand(uint64_t& s0, uint64_t& s1) {
  uint64_t n = (next_rand(s0, s1) >> 12) | 0x3FF0000000000000;
  auto* num = (double*)&n;
  return *num - 1.0;
}

// overload of operator<< for printing a vector
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  for (uint64_t i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i != vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

inline std::uint64_t customHash(std::uint64_t key) {
  key = (key ^ (key >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  key = (key ^ (key >> 27)) * UINT64_C(0x94d049bb133111eb);
  key = key ^ (key >> 31);
  return key;
}

// frozenset similar to Python class
template <typename T>
class FrozenSet {
private:
  std::vector<T> data;
  mutable uint64_t hash_value = 0;  // precomputed hash value

public:
  FrozenSet() : hash_value(0) {}

  FrozenSet(std::initializer_list<T> init_list) : data(init_list) {
    // sort the data
    std::sort(data.begin(), data.end());
    // remove duplicates
    data.erase(std::unique(data.begin(), data.end()), data.end());
  }

  // constructor with integer vector input
  FrozenSet(std::vector<T> input_vector) : data(std::move(input_vector)) {
    // sort the data
    std::sort(data.begin(), data.end());
    // remove duplicates
    data.erase(std::unique(data.begin(), data.end()), data.end());
  }

  FrozenSet(std::vector<T>&& input_vector, bool) : data(std::move(input_vector)) {}

  FrozenSet(const std::unordered_set<T>& set) : data(set.begin(), set.end()) {
    // sort the data
    std::sort(data.begin(), data.end());
    // remove duplicates
    data.erase(std::unique(data.begin(), data.end()), data.end());
  }

  FrozenSet(const ConsistentOrderedSet<T>& set) : data(set.begin(), set.end()) {
    // sort the data
    std::sort(data.begin(), data.end());
    // remove duplicates
    data.erase(std::unique(data.begin(), data.end()), data.end());
  }

  void calculateHash() const {
    for (const auto& item : data) {
      hash_value ^= customHash(item);
    }
  }

  bool contains(const T& value) const {
    // use binary search for sorted data
    return std::binary_search(data.begin(), data.end(), value);
  }

  uint64_t size() const { return data.size(); }

  bool empty() const { return data.empty(); }

  void print() const {
    std::cout << "{ ";
    for (const auto& item : data) {
      std::cout << item << " ";
    }
    std::cout << "}" << std::endl;
  }

  bool operator==(const FrozenSet& other) const { return data == other.data; }

  bool operator!=(const FrozenSet& other) const { return data != other.data; }

  uint64_t getHash() const {
    // lazy initialization of hash_value
    if (hash_value == 0) {
      calculateHash();
    }
    return hash_value;
  }

  const std::vector<T>& get_data() const { return data; }

  // efficient set union
  FrozenSet<T> operator|(const FrozenSet& other) const {
    FrozenSet<T> result;
    result.data.reserve(data.size() + other.data.size());

    auto it1 = data.begin();
    auto it2 = other.data.begin();

    while (it1 != data.end() && it2 != other.data.end()) {
      if (*it1 < *it2) {
        result.data.push_back(*it1);
        ++it1;
      } else if (*it1 > *it2) {
        result.data.push_back(*it2);
        ++it2;
      } else {
        // the elements are equal, add only one and advance both iterators
        result.data.push_back(*it1);
        ++it1;
        ++it2;
      }
    }

    // append the remaining elements
    result.data.insert(result.data.end(), it1, data.end());
    result.data.insert(result.data.end(), it2, other.data.end());

    return result;
  }

  // efficient set intersection
  FrozenSet<T> operator&(const FrozenSet& other) const {
    FrozenSet<T> result;
    result.data.reserve(std::min(data.size(), other.data.size()));
    auto it1 = data.begin();
    auto it2 = other.data.begin();

    while (it1 != data.end() && it2 != other.data.end()) {
      if (*it1 < *it2) {
        ++it1;
      } else if (*it1 > *it2) {
        ++it2;
      } else {
        // the elements are equal, add to the result and advance both iterators
        result.data.push_back(*it1);
        ++it1;
        ++it2;
      }
    }
    return result;
  }


  bool has_common(const FrozenSet &other) const {
    auto it1 = data.begin();
    auto it2 = other.data.begin();

    while (it1 != data.end() && it2 != other.data.end()) {
      if (*it1 == *it2)
        return true;
      else if (*it1 < *it2)
        ++it1;
      else
        ++it2;
    }
    return false;
  }


  bool all_common(const FrozenSet &other) const {
    auto it1 = data.begin();
    auto it2 = other.data.begin();

    while (it1 != data.end() && it2 != other.data.end()) {
      if (*it1 == *it2) {
        ++it2; // Move to the next element in other.data only if a match is found
      } else if (*it1 < *it2) {
        ++it1;
      } else {
        return false; // If an element in other.data is not found in data, return false
      }
    }

    // If we've checked all elements in other.data, return true
    return it2 == other.data.end();
  }

  // efficient set difference
  FrozenSet<T> operator-(const FrozenSet& other) const {
    FrozenSet<T> result;
    result.data.reserve(data.size());
    auto it1 = data.begin();
    auto it2 = other.data.begin();

    while (it1 != data.end() && it2 != other.data.end()) {
      if (*it1 < *it2) {
        // the element is present only in the first set, add to the result and advance the first iterator
        result.data.push_back(*it1);
        ++it1;
      } else if (*it1 > *it2) {
        // the element is not present in the second set, advance the second iterator
        ++it2;
      } else {
        // the elements are equal, advance both iterators to skip the common element
        ++it1;
        ++it2;
      }
    }

    // append the remaining elements from the first set
    result.data.insert(result.data.end(), it1, data.end());

    return result;
  }
};

namespace std {
  template <typename T>
  struct hash<FrozenSet<T>> {
  uint64_t operator()(const FrozenSet<T>& fs) const { return fs.getHash(); }
};
}  // namespace std

template<class T>
static inline void processChunk(const std::vector<T>& frozenSets, std::vector<T>& newFrozenSets,
                  const T& positions_set, uint64_t start, uint64_t end) {
  for (uint64_t i = start; i < end; ++i) {
    if (frozenSets[i].has_common(positions_set)) {
      newFrozenSets[i] = frozenSets[i] - positions_set;
    } else {
      newFrozenSets[i] = frozenSets[i];
    }
  }
}

template<class T>
void parallelProcessSets(const std::vector<T>& frozenSets, std::vector<T>& newFrozenSets,
                         const T& positions_set, int chunkSize) {
  const size_t totalSize = frozenSets.size();
  const size_t numChunks = (totalSize + chunkSize - 1) / chunkSize;

  std::vector<std::future<void>> futures;

  for (size_t chunk = 0; chunk < numChunks; ++chunk) {
    size_t start = chunk * chunkSize;
    size_t end = std::min(start + chunkSize, totalSize);

    futures.push_back(std::async(std::launch::async,
                                 [&frozenSets, &newFrozenSets, &positions_set, start, end]() {
                                   processChunk(frozenSets, newFrozenSets, positions_set, start, end);
                                 }));
  }
}

template <typename T>
class Expression {
protected:
  std::vector<double> index_to_size;
  std::vector<T> histogram;
  // mapping and index_to_size_map are only used for adding input vectors
  std::unordered_map<T, uint64_t> mapping;  // HashMap to map original numbers to new numbers
  std::unordered_map<T, uint64_t> index_to_size_map;
  std::vector<FrozenSet<T>> frozenSets;
public:
  Expression() = default;

  explicit Expression(std::unordered_map<T, uint64_t> index_to_size_map) : index_to_size_map(std::move(index_to_size_map)) {
    index_to_size.reserve(this->index_to_size_map.size());
    histogram.reserve(this->index_to_size_map.size());
  }

  void add(const std::vector<T>& input_vector) {
    std::vector<T> vfs;
    vfs.reserve(input_vector.size());
    for (const auto& num : input_vector) {
      if (mapping.find(num) == mapping.end()) {
        mapping[num] = mapping.size();  // Assign a new number for unseen elements
        if (index_to_size_map.find(num) == index_to_size_map.end()) {
          throw std::runtime_error("ERROR: Index '" + std::to_string(num) + "' is not part of the dictionary index_to_size.");
        } else {
          index_to_size.push_back(double(index_to_size_map[num]));
          histogram.push_back(0);
        }
      }
      vfs.push_back(mapping[num]);
    }
    frozenSets.emplace_back(std::move(vfs));
    for (T index : frozenSets.back().get_data()) {
      histogram[index]++;
    }
  }

  bool simplify(bool add_common_batch_to_output = true) {
    if (frozenSets.empty()) {
      return false;
    }
    // find positions in index_to_size where the values equal one
    std::vector<T> positions_to_remove;
    std::vector<T> common_batch_indices;
    for (uint64_t i = 0; i < histogram.size(); ++i) {
      if (histogram[i] == 1 || index_to_size[i] == 1.0) {
        positions_to_remove.push_back(T(i));
        histogram[i] = 0;
        index_to_size[i] = 0;
      } else if (histogram[i] == frozenSets.size() - 1 && !frozenSets.back().contains(i)) {
        if(add_common_batch_to_output) {
          common_batch_indices.push_back(i);
          histogram[i]++;
        }
      }
    }
    if (positions_to_remove.empty() && common_batch_indices.empty()) {
      return false;
    }
    const FrozenSet<T> positions_set(positions_to_remove);
    std::vector<FrozenSet<T>> newFrozenSets;
    newFrozenSets.resize(frozenSets.size());

    if (frozenSets.size() > 10000) {
      int chunkSize = 128;
      parallelProcessSets(frozenSets, newFrozenSets, positions_set, chunkSize);
    } else {
      for (uint64_t i = 0; i < frozenSets.size(); ++i) {
        if (frozenSets[i].has_common(positions_set)) {
          newFrozenSets[i] = frozenSets[i] - positions_set;
        } else {
          newFrozenSets[i] = frozenSets[i];
        }
      }
    }

    // Indices that are common to all tensors (common_batch_indices) might as well be output indices,
    // since they cannot be contracted until the final step. Here, we add the common_batch_indices
    // to the output indices, that is the last frozen set.
    newFrozenSets.back() = newFrozenSets.back() | common_batch_indices;
    frozenSets = std::move(newFrozenSets);
    return true;
  }

  void print() const {
    for (const auto& fs : frozenSets) {
      fs.print();
    }
    std::cout << index_to_size << std::endl;
    std::cout << histogram << std::endl;
  }

  const std::vector<double>& get_index_to_size() const { return index_to_size; }

  const std::vector<T>& get_histogram() const { return histogram; }

  const std::vector<FrozenSet<T>>& get_frozen_sets() const { return frozenSets; }

  const std::unordered_map<T, uint64_t>& get_mapping() const { return mapping; }

  std::unordered_map<T, uint64_t>& get_index_to_size_map() { return index_to_size_map; }

  uint64_t size() const { return frozenSets.size(); }

  const FrozenSet<T>& operator[](uint64_t index) const { return frozenSets[index]; }
};

struct Path_Pair {
  uint64_t a{};
  uint64_t b{};
  Path_Pair() = default;
  Path_Pair(uint64_t a, uint64_t b) {
    this->a = a;
    this->b = b;
    if (a > b) {
      std::swap(this->a, this->b);
    }
  }
};

struct Path {
  std::vector<Path_Pair> ssa;
  std::vector<Path_Pair> linear;
};

// overload of the << operator for Path_Pair
static std::ostream& operator<<(std::ostream& os, const Path_Pair& path_pair) {
  os << "(" << path_pair.a << ", " << path_pair.b << ")";
  return os;
}

// overload of the << operator for Path
static std::ostream& operator<<(std::ostream& os, const Path& path) {
  os << "ssa: [";
  for (uint64_t i = 0; i < path.ssa.size(); ++i) {
    os << path.ssa[i];
    if (i < path.ssa.size() - 1) {
      os << ", ";
    }
  }
  os << "]" << std::endl;

  os << "linear: [";
  for (uint64_t i = 0; i < path.linear.size(); ++i) {
    os << path.linear[i];
    if (i < path.linear.size() - 1) {
      os << ", ";
    }
  }
  os << "]" << std::endl;

  return os;
}

// overload of the << operator for FrozenSet
template <typename T>
std::ostream& operator<<(std::ostream& os, const FrozenSet<T>& fs) {
  os << "{ ";
  for (const auto& item : fs.get_data()) {
    os << item << " ";
  }
  os << "}";
  return os;
}

// overload of the << operator for std::unordered_map<FrozenSet<T>, uint64_t>
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_map<FrozenSet<T>, uint64_t>& map) {
  os << "{ ";
  for (const auto& pair : map) {
    os << pair.first << ": " << pair.second << ", ";
  }
  os << "}";
  return os;
}

// overload of the << operator for std::unordered_set<FrozenSet<T>>
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<FrozenSet<T>>& set) {
  os << "{ ";
  for (const auto& fs : set) {
    os << fs << ", ";
  }
  os << "}";
  return os;
}

// overload of the << operator for std::unordered_map<T, std::unordered_set<FrozenSet<T>>>
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_map<T, std::unordered_set<FrozenSet<T>>>& map) {
  os << "{ ";
  for (const auto& pair : map) {
    os << pair.first << ": " << pair.second << ", ";
  }
  os << "}";
  return os;
}

template <typename T>
double compute_size_by_dict(const FrozenSet<T>& indices, const std::vector<double>& idx_dict) {
  double ret = 1;
  for (const auto& i : indices.get_data()) {
    ret *= idx_dict[i];
  }
  return ret;
}

template <class T>
struct KernelParts {
  FrozenSet<T> contraction_dims;
  FrozenSet<T> k1_free_dims;
  FrozenSet<T> k2_free_dims;
  FrozenSet<T> batch_dims;

  KernelParts(const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2) {
    contraction_dims = k1 & k2;
    k1_free_dims = k1 - contraction_dims;
    k2_free_dims = k2 - contraction_dims;
    batch_dims = contraction_dims & k12;
    contraction_dims = contraction_dims - batch_dims;
  }
};

// here, you can implement your cost function, to enable it, go to 'greedy_exec'
template<typename T>
double cost_test(double size12, double size1, double size2, const FrozenSet<T> &k12, const FrozenSet<T> &k1,
                 const FrozenSet<T> &k2, double nTotal, uint64_t &s0, uint64_t &s1, const std::vector<double> &idx_dict,
                 const ConsistentOrderedMap<FrozenSet<T>, uint64_t> &remaining,
                 const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>> &dim_to_keys,
                 const std::vector<T> &k1_free_dims, const std::vector<T> &k2_free_dims,
                 const std::vector<T> &batch_dims,
                 const std::vector<T> &sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {
  return size12;
}

template<typename T>
double cost_batch_balanced(double size12, double size1, double size2, const FrozenSet<T> &k12, const FrozenSet<T> &k1,
                           const FrozenSet<T> &k2, double nTotal, uint64_t &s0, uint64_t &s1, const std::vector<double> &idx_dict,
                           const ConsistentOrderedMap<FrozenSet<T>, uint64_t> &remaining,
                           const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>> &dim_to_keys,
                           const std::vector<T> &k1_free_dims, const std::vector<T> &k2_free_dims,
                           const std::vector<T> &batch_dims,
                           const std::vector<T> &sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {

  {
    double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
    double cost = size12 - (size1 + size2) + next_double_rand(s0, s1) - (next_double_rand(s0, s1) * 0.01) * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;

    double dimension_adjustment = 0.0;
    for (const T& dim : k1_free_dims) {
      if (dim_to_keys.find(dim) != dim_to_keys.end()) {
        dimension_adjustment -= (log2(dim_to_keys[dim].size()) * log2(dim_to_keys[dim].size())) * 0.2;
      }
    }
    for (const T& dim : k2_free_dims) {
      if (dim_to_keys.find(dim) != dim_to_keys.end()) {
        dimension_adjustment -= (log2(dim_to_keys[dim].size()) * log2(dim_to_keys[dim].size())) * 0.2;
      }
    }

    for (const T& dim : k12.get_data()) {
      if (dim_to_keys.find(dim) != dim_to_keys.end()) {
        dimension_adjustment += (log2(dim_to_keys[dim].size()) * log2(dim_to_keys[dim].size())) * 0.2;
      }
    }

    cost += dimension_adjustment;

    const double cooling_probability = 0.2;
    double cost_boltzmann = 10 * std::max(size1, size2) - (next_double_rand(s0, s1)) * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;
    double temperature = double(size12 != 0.0) * (size12 + next_double_rand(s0, s1)) + 1 + 10e-12;
    double boltzmann_weight = std::exp(-cost_boltzmann / (temperature));
    return cooling_probability * cost / (size12 + 10e-12) + boltzmann_weight * (1.0 - cooling_probability);
  }
}

// the basic randomized cost function
template <typename T>
double cost_memory_removed_jitter(double size12, double size1, double size2, const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2, double nTotal, uint64_t& s0, uint64_t& s1, const std::vector<double>& idx_dict, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& k1_free_dims, const std::vector<T>& k2_free_dims, const std::vector<T>& batch_dims,
                                  const std::vector<T>& sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {
  double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
  double cost = size12 - global_alpha * 50 * (size1 + size2) - next_double_rand(s0, s1) * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;
  cost += (next_double_rand(s0, s1) - 0.5) * 0.1;

  return cost;
}

// works sometimes good when the tensor network, has no hyper edges that are part of other hyper edges
template <typename T>
double cost_boltzmann(double size12, double size1, double size2, const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2, double nTotal, uint64_t& s0, uint64_t& s1, const std::vector<double>& idx_dict, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& k1_free_dims, const std::vector<T>& k2_free_dims, const std::vector<T>& batch_dims,
                      const std::vector<T>& sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {
  double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
  double alpha = 0.8 + 0.2 * next_double_rand(s0, s1);
  double cost = size12 + alpha * (size1 + size2) + std::max(size1, size2) - next_double_rand(s0, s1) * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;
  double temperature = double(size12 != 0.0) * (size12 + next_double_rand(s0, s1)) + 10e-12;
  double boltzmann_weight = std::exp(-cost / temperature);
  return boltzmann_weight;
}

template <typename T>
double cost_skew_balanced(double size12, double size1, double size2, const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2, double nTotal, uint64_t& s0, uint64_t& s1, const std::vector<double>& idx_dict, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& k1_free_dims, const std::vector<T>& k2_free_dims, const std::vector<T>& batch_dims,
                          const std::vector<T>& sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {
  double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
  double cost = size12 - (size1 + size2);
  cost += std::abs(size1 - size2) * 0.15;
  cost += next_double_rand(s0, s1) - next_double_rand(s0, s1)  * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;

  double dimension_adjustment = 0.0;
  for (const T& dim : k1_free_dims) {
    if (dim_to_keys.find(dim) != dim_to_keys.end()) {
      dimension_adjustment -= dim_to_keys[dim].size() * 0.15;
    }
  }
  for (const T& dim : k2_free_dims) {
    if (dim_to_keys.find(dim) != dim_to_keys.end()) {
      dimension_adjustment -= dim_to_keys[dim].size() * 0.15;
    }
  }
  cost += dimension_adjustment;
  return cost / (size12 + next_double_rand(s0, s1) + 10e-12);
}

template <typename T>
double cost_balanced_boltzmann(double size12, double size1, double size2, const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2, double nTotal, uint64_t& s0, uint64_t& s1, const std::vector<double>& idx_dict, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& k1_free_dims, const std::vector<T>& k2_free_dims, const std::vector<T>& batch_dims,
                               const std::vector<T>& sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {
  double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
  double cost = size12 - (size1 + size2) + next_double_rand(s0, s1) - next_double_rand(s0, s1) * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;

  // promote outer dimensions when they belong to hyperedges shared among many tensors
  // in this way, small hyperedges should be removed first and the sizes of the hyperedges should be balanced
  double dimension_adjustment = 0.0;
  for (const T& dim : k1_free_dims) {
    if (dim_to_keys.find(dim) != dim_to_keys.end()) {
      dimension_adjustment -= dim_to_keys[dim].size() * 0.15;
    }
  }
  for (const T& dim : k2_free_dims) {
    if (dim_to_keys.find(dim) != dim_to_keys.end()) {
      dimension_adjustment -= dim_to_keys[dim].size() * 0.15;
    }
  }
  cost += dimension_adjustment;

//  cost += size12 * (next_double_rand(s0, s1)) * 0.15 - std::abs(size1 - size2);

  const double cooling_probability = 0.2;
  double alpha = 0.8 + 0.2 * next_double_rand(s0, s1);
  double cost_boltzmann = size12 + alpha * (size1 + size2) + std::max(size1, size2) - next_double_rand(s0, s1) * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;;
  double temperature = double(size12 != 0.0) * (size12 + next_double_rand(s0, s1)) + 10e-12;
  double boltzmann_weight = std::exp(-cost_boltzmann / temperature);
  return cooling_probability * cost / (size12 + 10e-12) + boltzmann_weight * (1.0 - cooling_probability);
}

template <typename T>
double cost_max_skew(double size12, double size1, double size2, const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2, double nTotal, uint64_t& s0, uint64_t& s1, const std::vector<double>& idx_dict, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& k1_free_dims, const std::vector<T>& k2_free_dims, const std::vector<T>& batch_dims,
                     const std::vector<T>& sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {
  double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
  double cost = size12 - next_double_rand(s0, s1) * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;
  double alpha = (nTotal - remaining.size()) / nTotal;
  cost += alpha * size12 * next_double_rand(s0, s1);
  cost = cost / ((1.0 - alpha) * std::max(size1, size2) * std::abs(size1 - size2) + next_double_rand(s0, s1) + size1 * size2 + 10e-12);
  return cost;
}

template <typename T>
double cost_anti_balanced(double size12, double size1, double size2, const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2, double nTotal, uint64_t& s0, uint64_t& s1, const std::vector<double>& idx_dict, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& k1_free_dims, const std::vector<T>& k2_free_dims, const std::vector<T>& batch_dims,
                          const std::vector<T>& sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {
  double cost = size12 - (size1 + size2);
  cost -= std::abs(size1 - size2) * 0.15;
  cost -= std::max(size1, size2) * 0.15;
  //  double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
  double cooling = 1.0 - (nTotal - remaining.size()) / nTotal;
  cost += next_double_rand(s0, s1) - next_double_rand(s0, s1)  * ((log2(size2 + 2) + log2(size1 + 2)) / log2(size12 + 2)) / cooling;

  // promote outer dimensions when they belong to hyperedges shared among many tensors
  // in this way, small hyperedges should be removed first and the sizes of the hyperedges should be balanced
  double dimension_adjustment = 0.0;
  for (const T& dim : k1_free_dims) {
    if (dim_to_keys.find(dim) != dim_to_keys.end()) {
      dimension_adjustment -= dim_to_keys[dim].size() * 0.15;
    }
  }
  for (const T& dim : k2_free_dims) {
    if (dim_to_keys.find(dim) != dim_to_keys.end()) {
      dimension_adjustment -= dim_to_keys[dim].size() * 0.15;
    }
  }
  cost -= dimension_adjustment;
  return cost / (size12 + next_double_rand(s0, s1) + 10e-12);
}

template <typename T>
double cost_log(double size12, double size1, double size2, const FrozenSet<T>& k12, const FrozenSet<T>& k1, const FrozenSet<T>& k2, double nTotal, uint64_t& s0, uint64_t& s1, const std::vector<double>& idx_dict, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& k1_free_dims, const std::vector<T>& k2_free_dims, const std::vector<T>& batch_dims,
                const std::vector<T>& sum_dims, const FrozenSet<T>& global_k, const double global_alpha) {

  return log2(size12 + 2) / log2((size1 + size2) * 0.65 + 2 + next_double_rand(s0, s1));
}

template <class T>
struct CandidateContraction {
  double cost;
  Path_Pair pp;
  FrozenSet<T> k1;
  FrozenSet<T> k2;
  FrozenSet<T> k12;

  CandidateContraction(double c, Path_Pair pp, const FrozenSet<T>& k1, const FrozenSet<T>& k2, const FrozenSet<T>& k12) : cost(c), pp(pp.a, pp.b), k1(k1), k2(k2), k12(k12) {}

  // define a custom comparison function for sorting in the priority queue
  bool operator<(const CandidateContraction& other) const { return cost > other.cost; }
};

template <class T>
std::ostream& operator<<(std::ostream& os, const CandidateContraction<T>& candidate) {
  os << "Cost: " << candidate.cost << ", pp: " << candidate.pp << ", k1: " << candidate.k1 << ", k2: " << candidate.k2 << ", k12: " << candidate.k12;
  return os;
}

template <class T>
void print_priority_queue(const std::priority_queue<CandidateContraction<T>>& queue) {
  std::priority_queue<CandidateContraction<T>> tempQueue = queue;  // Make a copy of the queue
  while (!tempQueue.empty()) {
    const CandidateContraction<T>& candidate = tempQueue.top();
    std::cout << candidate << std::endl;  // Assuming you've already overloaded operator<< for CandidateContraction<T>
    tempQueue.pop();
  }
}

template <class T>
using CostFunctionType = double (*)(double, double, double, const FrozenSet<T>&, const FrozenSet<T>&, const FrozenSet<T>&, double, uint64_t&, uint64_t&, const std::vector<double>&, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>&, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&, const FrozenSet<T>&, const double);

template <class T>
void push_candidate(const FrozenSet<T>& output, const std::vector<double>& sizes, ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, const FrozenSet<T>& k1, const ConsistentOrderedSet<FrozenSet<T>>& k2s, std::priority_queue<CandidateContraction<T>>& queue, CostFunctionType<T> cost_fn, uint64_t& s0, uint64_t& s1, double nTotal, const ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>>& dim_to_keys, const std::vector<T>& histogram, std::vector<T>& k1_free_dims,
                    std::vector<T>& k2_free_dims, std::vector<T>& batch_dims, std::vector<T>& sum_dims, const double ident_size, const FrozenSet<T>& k_global, const double alpha) {
  double size1;
  if (ident_size == -1.0) {
    size1 = compute_size_by_dict(k1, sizes);
  } else {
    size1 = std::pow(ident_size, double(k1.get_data().size()));
  }
  for (const auto& k2 : k2s) {
    const auto k12 = compute_k12_const(k1, k2, histogram, k1_free_dims, k2_free_dims, batch_dims, sum_dims);

    double size12;
    double size2;
    if (ident_size == -1.0) {
      size12 = compute_size_by_dict(k12, sizes);
      size2 = compute_size_by_dict(k2, sizes);
    } else {
      size12 = std::pow(ident_size, double(k12.get_data().size()));
      size2 = std::pow(ident_size, double(k2.get_data().size()));
    }

    double cost = cost_fn(size12, size1, size2, k12, k1, k2, nTotal, s0, s1, sizes, remaining, dim_to_keys, k1_free_dims, k2_free_dims, batch_dims, sum_dims, k_global, alpha);

    // hadamard first with higher probability if no mm or bmm
    if (remaining.find(k12) != remaining.end() && !((!k1_free_dims.empty() && !k2_free_dims.empty() && !sum_dims.empty()))) {
      cost -= std::abs(cost) * next_double_rand(s0, s1);
    }

    uint64_t id1 = remaining[k1];
    uint64_t id2 = remaining[k2];
    CandidateContraction<T> candidate(cost, Path_Pair{id1, id2}, k1, k2, std::move(k12));
    if (id1 > id2) {
      std::swap(candidate.k1, candidate.k2);
    }
    queue.push(std::move(candidate));
  }
}

template <typename T>
std::optional<std::tuple<double, FrozenSet<T>, FrozenSet<T>, FrozenSet<T>>> simple_chooser(std::priority_queue<CandidateContraction<T>>& queue, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining) {
  if (queue.empty()) {
    return std::nullopt;
  }

  CandidateContraction<T> candidate = queue.top();
  queue.pop();

  const FrozenSet<T>& k1 = candidate.k1;
  const FrozenSet<T>& k2 = candidate.k2;

  if (remaining.find(k1) == remaining.end() || remaining.find(k2) == remaining.end()) {
    return std::nullopt;
  }

  return std::make_tuple(candidate.cost, k1, k2, candidate.k12);
}

template <typename T>
std::optional<std::tuple<double, FrozenSet<T>, FrozenSet<T>, FrozenSet<T>>> thermal_chooser(std::priority_queue<CandidateContraction<T>>& queue, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, uint64_t& s0, uint64_t& s1, int nbranch = 8, double temperature = 1.0, bool rel_temperature = true) {
  int n = 0;
  std::vector<CandidateContraction<T>> choices;

  while (!queue.empty() && n < nbranch) {
    CandidateContraction<T> candidate = queue.top();
    queue.pop();

    const FrozenSet<T>& k1 = candidate.k1;
    const FrozenSet<T>& k2 = candidate.k2;

    if (remaining.find(k1) == remaining.end() || remaining.find(k2) == remaining.end()) {
      continue;  // candidate is obsolete
    }

    choices.push_back(candidate);
    n++;
  }

  if (n == 0) {
    return std::nullopt;
  }
  if (n == 1) {
    return std::make_tuple(choices[0].cost, choices[0].k1, choices[0].k2, choices[0].k12);
  }

  std::vector<double> costs;
  for (const auto& choice : choices) {
    costs.push_back(choice.cost);
  }
  double cmin = costs[0];

  if (rel_temperature) {
    temperature *= std::max(1.0, std::abs(cmin));
  }

  std::vector<double> energies;
  if (temperature == 0.0) {
    for (const auto& c : costs) {
      energies.push_back(c == cmin ? 1.0 : 0.0);
    }
  } else {
    for (const auto& c : costs) {
      energies.push_back(std::exp(-(c - cmin) / temperature));
    }
  }

  // calculate cumulative distribution
  std::vector<double> cum_energies;
  double sum = 0.0;
  for (double e : energies) {
    sum += e;
    cum_energies.push_back(sum);
  }

  // use next_double_rand to choose based on energies
  double rand_value = next_double_rand(s0, s1) * sum;
  uint64_t chosen = std::lower_bound(cum_energies.begin(), cum_energies.end(), rand_value) - cum_energies.begin();

  auto choice = choices[chosen];
  choices.erase(choices.begin() + chosen);

  // push other choices back into the priority queue
  for (const auto& other : choices) {
    queue.push(other);
  }

  return std::make_tuple(choice.cost, choice.k1, choice.k2, choice.k12);
}


template<uint64_t Elements>
struct Id_Tracker {
  std::vector<std::bitset<Elements>> buckets;
  std::vector<uint64_t> ones_per_bucket;

  explicit Id_Tracker(uint64_t num_ids) : buckets(num_ids / Elements + 1),
                                          ones_per_bucket(num_ids / Elements + 1) {}

  void set_id_to_one(uint64_t id) {
    uint64_t bucket = id / Elements;
    uint64_t idx = id % Elements;
    buckets[bucket][idx] = 1;
    ones_per_bucket[bucket]++;
  }

  uint64_t get_linear_id(uint64_t id) {
    uint64_t bucket = id / Elements;
    uint64_t idx = id % Elements;
    uint64_t num_ones = 0;
    for (uint64_t i = 0; i < bucket; ++i) {
      num_ones += ones_per_bucket[i];
    }

    std::bitset<Elements> mask;
    mask.set();
    mask >>= (Elements - idx - 1);

    num_ones += (buckets[bucket] & mask).count();
    return id - num_ones;
  }
};

struct SSA_Node {
  uint64_t left;
  uint64_t right;
  uint64_t value;
};

static void
traverse_ssa_path_tree(std::vector<SSA_Node>&ssa_tree, uint64_t id, std::vector<Path_Pair> &new_ssa_path, uint64_t &free_id, uint64_t free_id_start) {
  std::vector<uint64_t> s;
  uint64_t last_node_visited = std::numeric_limits<uint64_t>::max();
  while (!s.empty() || id >= free_id_start) {
    if (id >= free_id_start) {
      s.push_back(id);
      id = ssa_tree[id].left;
    } else {
      uint64_t top_id = s.back();
      if (ssa_tree[top_id].right >= free_id_start && last_node_visited != ssa_tree[top_id].right) {
        id = ssa_tree[top_id].right;
      } else {
        uint64_t a = ssa_tree[ssa_tree[top_id].left].value;
        uint64_t b = ssa_tree[ssa_tree[top_id].right].value;
        new_ssa_path.emplace_back(a, b);
        ssa_tree[top_id].value = free_id;
        free_id++;
        last_node_visited = s.back();
        s.pop_back();
      }
    }
  }
}

static std::vector<Path_Pair> ssa_to_post_order_ssa(const std::vector<Path_Pair>& ssa_path) {
  if(ssa_path.size() < 2) {
    return ssa_path;
  }
  uint64_t free_id = ssa_path.size() + 1;
  std::vector<SSA_Node> ssa_tree(free_id + free_id - 1);
  for (uint64_t i = 0; i < free_id; ++i) {
    ssa_tree[i].value = i;
  }
  for(const auto& pp: ssa_path) {
    ssa_tree[free_id].left = pp.a;
    ssa_tree[free_id].right = pp.b;
    free_id++;
  }
  free_id = ssa_path.size() + 1;
  std::vector<Path_Pair> new_ssa_path;
  new_ssa_path.reserve(ssa_path.size());
  traverse_ssa_path_tree(ssa_tree, ssa_tree.size() - 1, new_ssa_path, free_id, free_id);
  return new_ssa_path;
}


static std::vector<Path_Pair> ssa_to_linear(const std::vector<Path_Pair>& ssa_path) {
  uint64_t max_id = 0;
  for (const auto& ids : ssa_path) {
    max_id = std::max({max_id, ids.a, ids.b});
  }
  std::vector<uint64_t> ids(max_id + 1);
  for (uint64_t i = 0; i <= max_id; ++i) {
    ids[i] = i;
  }

  Id_Tracker<2048> id_tracker(max_id + 1);

  std::vector<Path_Pair> path;
  for (const auto& ssa_ids : ssa_path) {
    path.emplace_back(id_tracker.get_linear_id(ssa_ids.a), id_tracker.get_linear_id(ssa_ids.b));
    id_tracker.set_id_to_one(ssa_ids.a);
    id_tracker.set_id_to_one(ssa_ids.b);
  }
  return path;
}

template <typename Key, typename T>
static std::ostream& operator<<(std::ostream& os, const ConsistentOrderedMap<Key, T>& map) {
  os << "{ ";
  for (const auto& pair : map) {
    os << pair.first << ": " << pair.second << ", ";
  }
  os << "}";
  return os;
}

template <class T>
FrozenSet<T> compute_k12(const FrozenSet<T>& k1, const FrozenSet<T>& k2, std::vector<T>& histogram) {
  FrozenSet<T> _k12 = k1 | k2;
  std::vector<T> to_remove;
  for (auto i : k1.get_data()) {
    --histogram[i];
    if (histogram[i] == 0) {
      to_remove.push_back(i);
    }
  }
  for (auto i : k2.get_data()) {
    --histogram[i];
    if (histogram[i] == 0) {
      to_remove.push_back(i);
    }
  }
  FrozenSet<T> k12 = _k12 - to_remove;
  for (auto i : k12.get_data()) {
    histogram[i]++;
  }
  return k12;
}

template <class T>
inline FrozenSet<T> compute_k12_const(const FrozenSet<T>& k1, const FrozenSet<T>& k2, const std::vector<T>& histogram, std::vector<T>& k1_free_dims, std::vector<T>& k2_free_dims, std::vector<T>& batch_dims, std::vector<T>& sum_dims) {
  k1_free_dims.clear();
  k2_free_dims.clear();
  batch_dims.clear();
  sum_dims.clear();
  const auto& vec1 = k1.get_data();
  const auto& vec2 = k2.get_data();
  std::vector<T> final;
  final.reserve(vec1.size() + vec2.size());
  uint64_t i = 0, j = 0;
  while (i < vec1.size() && j < vec2.size()) {
    if (vec1[i] < vec2[j]) {
      final.push_back(vec1[i]);
      k1_free_dims.push_back(vec1[i]);
      ++i;
    } else if (vec1[i] > vec2[j]) {
      final.push_back(vec2[j]);
      k2_free_dims.push_back(vec2[j]);
      ++j;
    } else {
      if (histogram[vec1[i]] != 2) {
        sum_dims.push_back(vec1[i]);
        final.push_back(vec1[i]);
      } else {
        batch_dims.push_back(vec1[i]);
      }
      ++i;
      ++j;
    }
  }
  while (i < vec1.size()) {
    final.push_back(vec1[i]);
    ++i;
  }
  while (j < vec2.size()) {
    final.push_back(vec2[j]);
    ++j;
  }

  return FrozenSet<T>(std::move(final), true);
}

struct Metrics {
  double log10_flops;
  double max_log2_size;
  double flops;
  double max_size;

  Metrics(double log10Flops, double maxLog2Size, double flops, double maxSize) : log10_flops(log10Flops),
                                                                                 max_log2_size(maxLog2Size),
                                                                                 flops(flops), max_size(maxSize) {}

  Metrics() : log10_flops(std::numeric_limits<double>::max()), max_log2_size(std::numeric_limits<double>::max()),
              flops(std::numeric_limits<double>::max()), max_size(std::numeric_limits<double>::max()) {}

  friend bool operator==(const Metrics& lhs, const Metrics& rhs) {
    // Consider two Metrics objects equal if all their attributes are exactly the same
    // Use std::fabs to compare floating-point numbers for equality within a small tolerance
    double tolerance = std::numeric_limits<double>::epsilon();
    return std::fabs(lhs.log10_flops - rhs.log10_flops) < tolerance &&
           std::fabs(lhs.max_log2_size - rhs.max_log2_size) < tolerance &&
           std::fabs(lhs.flops - rhs.flops) < tolerance &&
           std::fabs(lhs.max_size - rhs.max_size) < tolerance;
  }

  friend bool operator!=(const Metrics& lhs, const Metrics& rhs) {
    // Use the already defined == operator for Metrics
    return !(lhs == rhs);
  }

  // define the operator<< as a friend function
  friend std::ostream& operator<<(std::ostream& os, const Metrics& metrics) {
    os << "Metrics: { log10_flops = " << metrics.log10_flops << ", max_log2_size = " << metrics.max_log2_size << " }";
    return os;
  }
};

template <class T>
void update_metrics(const std::vector<double>& sizes, std::vector<T>& histogram, std::vector<FrozenSet<T>>& inputs, uint64_t& i_current, uint64_t ssa_id1, uint64_t ssa_id2, double& max_size, double& flops) {
  const auto& k1 = inputs[ssa_id1];
  const auto& k2 = inputs[ssa_id2];
  const auto k12 = compute_k12<T>(k1, k2, histogram);
  flops += compute_size_by_dict<T>(k1 | k2, sizes) * 2;
  max_size = std::max(compute_size_by_dict<T>(k12, sizes), max_size);
  inputs[i_current] = std::move(k12);
  i_current++;
}

template <class T>
Metrics compute_metrics(const Expression<T>& expression, const Path& path) {
  std::vector<T> histogram = expression.get_histogram();
  const auto& sizes = expression.get_index_to_size();

  double max_size = 0;
  double flops = 0;

  std::vector<FrozenSet<T>> inputs(expression.size() + expression.size() - 1);
  for (uint64_t i = 0; i < expression.size() - 1; ++i) {
    inputs[i] = expression[i];
  }
  uint64_t i = expression.size() - 1;
  for (const auto& contraction : path.ssa) {
    update_metrics(sizes, histogram, inputs, i, contraction.a, contraction.b, max_size, flops);
  }

  return {log10(flops), log2(max_size), flops, max_size};
}

struct ResultInner {
  Path path;
  double flops;
  double max_size;
};

enum Minimize { FLOPS, INTERMEDIATE_SIZE };

static inline bool are_equal(double a, double b, double epsilon = 1e-8) { return std::abs(a - b) < epsilon; }

static inline bool flops_minimize(double flops_current, double flops_best, double size_current, double size_best) {
  if (are_equal(flops_current, flops_best)) {
    return size_current < size_best;
  }
  return flops_current < flops_best;
}

static inline bool size_minimize(double flops_current, double flops_best, double size_current, double size_best) { return flops_minimize(size_current, size_best, flops_current, flops_best); }

using MinimizeFuncType = bool (*)(double, double, double, double);

struct GreedyResult {
  Path path;
  Metrics metrics{};
};

/************** dynamic programming optimal **************/

template <typename T>
inline void unionSortedVectorsInPlace(const std::vector<T>& vec1, const std::vector<T>& vec2, std::vector<T>& result) {
  result.resize(vec1.size() + vec2.size());
  auto it = std::set_union(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
  result.resize(it - result.begin());
}

template <class T>
std::vector<T> recursiveMerge(const std::vector<FrozenSet<T>>& inputs, uint64_t bitmap, T start, T end) {
  // base condition adjusted: if the range is invalid or there's no bit set in the range, return empty.
  if (start > end) return {};

  // if there's only one element left, check if its corresponding bit is set and return it if so
  if (start == end) {
    if (bitmap & (1ull << start)) {
      return inputs[start].get_data(); // return the data if the bit is set.
    } else {
      return {}; // return empty if the bit is not set.
    }
  }

  // recursively process the left and right halves of the range.
  T mid = start + (end - start) / 2;
  std::vector<T> left = recursiveMerge(inputs, bitmap, start, mid);
  std::vector<T> right = recursiveMerge(inputs, bitmap, mid + 1, end);
  std::vector<T> result;

  // Merge the results from the left and right.
  unionSortedVectorsInPlace(left, right, result);
  return result;
}


template <class T>
FrozenSet<T> bitmap_select(uint64_t bitmap, const std::vector<FrozenSet<T>>& inputs) {
  T start = 0;
  T end = inputs.size() - 1;
  return recursiveMerge(inputs, bitmap, start, end);
}

template <typename T>
struct TreeNode {
  T value;
  std::shared_ptr<TreeNode<T>> left;
  std::shared_ptr<TreeNode<T>> right;

  // default constructor
  TreeNode() : value(std::numeric_limits<T>::max()), left(nullptr), right(nullptr) {}

  // value constructor
  TreeNode(T val) : value(val), left(nullptr), right(nullptr) {}

  // value and child nodes constructor
  TreeNode(T val, std::shared_ptr<TreeNode<T>> leftNode, std::shared_ptr<TreeNode<T>> rightNode) : value(val), left(leftNode), right(rightNode) {}

  // copy constructor
  TreeNode(const TreeNode<T>& other) : value(other.value), left(other.left ? std::make_shared<TreeNode<T>>(*other.left) : nullptr), right(other.right ? std::make_shared<TreeNode<T>>(*other.right) : nullptr) {}

  // move constructor
  TreeNode(TreeNode<T>&& other) noexcept : value(std::move(other.value)), left(std::move(other.left)), right(std::move(other.right)) {}

  // copy assignment operator
  TreeNode<T>& operator=(const TreeNode<T>& other) {
    if (this != &other) {  // protection against self-assignment
      value = other.value;
      left = other.left ? std::make_shared<TreeNode<T>>(*other.left) : nullptr;
      right = other.right ? std::make_shared<TreeNode<T>>(*other.right) : nullptr;
    }
    return *this;
  }

  // move assignment operator
  TreeNode<T>& operator=(TreeNode<T>&& other) noexcept {
    if (this != &other) {  // protection against self-assignment
      value = std::move(other.value);
      left = std::move(other.left);
      right = std::move(other.right);
    }
    return *this;
  }
};

template <typename T>
void generatePathPairsFromPostorder(const std::shared_ptr<TreeNode<T>>& root, std::vector<Path_Pair>& pathPairs, uint64_t& free_id) {
  if (!root) return;
  // traverse left subtree
  generatePathPairsFromPostorder(root->left, pathPairs, free_id);
  // traverse right subtree
  generatePathPairsFromPostorder(root->right, pathPairs, free_id);
  // only add pairs when there are child nodes (i.e., not for leaf nodes)
  if (root->left && root->right) {
    uint64_t a = root->left->value;
    uint64_t b = root->right->value;
    pathPairs.emplace_back(a, b);
    root->value = free_id;
    free_id++;
  }
}

template <class T>
using TensorSet = FrozenSet<T>;
template <class T>
using DPEntry = std::tuple<TensorSet<T>, double, double, std::shared_ptr<TreeNode<T>>>;
template <class T>
using CheckContractionType = void (*)(double, double, double, double, const FrozenSet<T>&, const std::vector<double>&, double, uint64_t, uint64_t, std::unordered_map<uint64_t, DPEntry<T>>&, uint64_t, uint64_t, const std::vector<FrozenSet<T>>&, FrozenSet<T>, const std::shared_ptr<TreeNode<T>>& cntrct1, const std::shared_ptr<TreeNode<T>>& cntrct2);

template <class T>
bool check_outer_false(const FrozenSet<T>& x) {
  return !x.empty();
}

template <class T>
bool check_outer_true(const FrozenSet<T>& x) {
  return true;
}

template <class T>
FrozenSet<T> dp_calc_legs(uint64_t g, uint64_t all_tensors, uint64_t s, const std::vector<FrozenSet<T>>& inputs, const FrozenSet<T>& i1_cut_i2_wo_output, const FrozenSet<T>& i1_union_i2) {
  // set of remaining tensors (=g-s)
  uint64_t r = g & (all_tensors ^ s);
  // indices of remaining indices:
  FrozenSet<T> i_r;
  if (r) {
    i_r = bitmap_select<T>(r, inputs);
  }
  // contraction indices:
  FrozenSet<T> i_contract = i1_cut_i2_wo_output - i_r;
  return i1_union_i2 - i_contract;
}

inline bool approximatelyLessThanOrEqual(double a, double b, double epsilon = 1e-8) { return a <= b || (a - b) < epsilon; }

template <class T>
void dp_compare_flops(double cost1, double cost2, double cost1_second, double cost2_second, const FrozenSet<T>& i1_union_i2, const std::vector<double>& size_dict, double cost_cap, uint64_t s1, uint64_t s2, std::unordered_map<uint64_t, DPEntry<T>>& xn, uint64_t g_bitmap, uint64_t all_tensors, const std::vector<FrozenSet<T>>& inputs, FrozenSet<T> i1_cut_i2_wo_output, const std::shared_ptr<TreeNode<T>>& cntrct1, const std::shared_ptr<TreeNode<T>>& cntrct2) {
  // Compute the cost
  double size = compute_size_by_dict(i1_union_i2, size_dict);
  double cost = static_cast<double>(cost1) + static_cast<double>(cost2) + size;
  if (approximatelyLessThanOrEqual(cost, cost_cap, 1e-8)) {
    uint64_t s = s1 | s2;
    if (xn.find(s) == xn.end()) {
      FrozenSet<T> i = dp_calc_legs(g_bitmap, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2);
      double mem = compute_size_by_dict(i, size_dict);
      double cost_second = std::max({static_cast<double>(cost1_second), static_cast<double>(cost2_second), mem});
      auto combined = std::make_shared<TreeNode<T>>(std::numeric_limits<T>::max(), cntrct1, cntrct2);
      xn[s] = std::make_tuple(i, cost, cost_second, combined);
    } else if (approximatelyLessThanOrEqual(cost, std::get<1>(xn[s]), 1e-8)) {
      FrozenSet<T> i = dp_calc_legs(g_bitmap, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2);
      double mem = compute_size_by_dict(i, size_dict);
      double cost_second = std::max({static_cast<double>(cost1_second), static_cast<double>(cost2_second), mem});
      if (approximatelyLessThanOrEqual(cost_second, std::get<2>(xn[s]), 1e-8)) {
        auto combined = std::make_shared<TreeNode<T>>(std::numeric_limits<T>::max(), cntrct1, cntrct2);
        xn[s] = std::make_tuple(i, cost, cost_second, combined);
      }
    }
  }
}

template <class T>
void dp_compare_size(double cost1, double cost2, double cost1_second, double cost2_second, const FrozenSet<T>& i1_union_i2, const std::vector<double>& size_dict, double cost_cap, uint64_t s1, uint64_t s2, std::unordered_map<uint64_t, DPEntry<T>>& xn, uint64_t g_bitmap, uint64_t all_tensors, const std::vector<FrozenSet<T>>& inputs, FrozenSet<T> i1_cut_i2_wo_output, const std::shared_ptr<TreeNode<T>>& cntrct1, const std::shared_ptr<TreeNode<T>>& cntrct2) {
  uint64_t s = s1 | s2;
  FrozenSet<T> i = dp_calc_legs(g_bitmap, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2);
  double mem = compute_size_by_dict(i, size_dict);
  double cost = std::max({static_cast<double>(cost1), static_cast<double>(cost2), mem});

  if (approximatelyLessThanOrEqual(cost, cost_cap)) {
    double cost_second = static_cast<double>(cost1_second) + static_cast<double>(cost2_second) + compute_size_by_dict(i1_union_i2, size_dict);
    if (xn.find(s) == xn.end() || (approximatelyLessThanOrEqual(cost, std::get<1>(xn[s])) && cost_second < std::get<2>(xn[s]))) {
      auto combined = std::make_shared<TreeNode<T>>(std::numeric_limits<T>::max(), cntrct1, cntrct2);
      xn[s] = std::make_tuple(i, cost, cost_second, combined);
    }
  }
}

template <class T>
void dynamic_result(const std::vector<std::unordered_map<uint64_t, DPEntry<T>>>& x, std::vector<std::shared_ptr<TreeNode<T>>>& subgraph_contractions, std::vector<double>& subgraph_contractions_size, const std::vector<double>& size_dict) {
  if (x.back().size() == 0) {
    return;
  }
  auto last_map = x.back();
  auto first_entry = last_map.begin();
  auto [i, cost, cost_second, contraction] = first_entry->second;
  subgraph_contractions.push_back(contraction);
  subgraph_contractions_size.push_back(compute_size_by_dict(i, size_dict));
}

template <class T>
std::vector<std::unordered_set<uint64_t>> find_disconnected_subgraphs(const std::vector<FrozenSet<T>>& inputs, const FrozenSet<T>& output) {
  std::vector<std::unordered_set<uint64_t>> subgraphs;
  std::unordered_set<uint64_t> unused_inputs;
  for (uint64_t i = 0; i < inputs.size(); ++i) {
    unused_inputs.insert(i);
  }
  FrozenSet<T> i_sum;  // summation indices
  for (const auto& input : inputs) {
    i_sum = i_sum | input;  // set union
  }
  i_sum = i_sum - output;  // set difference
  while (!unused_inputs.empty()) {
    std::unordered_set<uint64_t> g;
    std::vector<uint64_t> q;
    auto it = unused_inputs.begin();
    q.push_back(*it);
    unused_inputs.erase(it);
    while (!q.empty()) {
      uint64_t j = q.back();
      q.pop_back();
      g.insert(j);
      FrozenSet<T> i_tmp = i_sum & inputs[j];
      std::vector<uint64_t> to_remove;
      for (const auto& k : unused_inputs) {
        if (i_tmp.has_common(inputs[k])) {
          q.push_back(k);
          to_remove.push_back(k);
        }
      }
      for (const auto& r : to_remove) {
        unused_inputs.erase(r);
      }
    }
    subgraphs.push_back(g);
  }
  return subgraphs;
}

template <class T>
GreedyResult dynamic(const Expression<T>& expression, Minimize minimize = Minimize::INTERMEDIATE_SIZE, bool is_cost_cap = true, bool search_outer = false) {
  Path path;
  const auto sizes = expression.get_index_to_size();
  if (expression.size() < 2) {
    throw std::runtime_error("ERROR: Expression contains less than two tensors.");
  } else if (expression.size() == 2) {
    path.ssa.emplace_back(0, 0);
    path.linear.emplace_back(0, 0);
    double _size = compute_size_by_dict<T>(expression[0], sizes);
    return GreedyResult{path, Metrics{log10(_size * 2), log2(_size), _size * 2, _size}};
  } else if (expression.size() == 3) {
    path.ssa.emplace_back(0, 1);
    path.linear.emplace_back(0, 1);
    const auto& k1 = expression[0];
    const auto& k2 = expression[1];
    const auto& k12 = expression[2];
    double flops = compute_size_by_dict<T>(k1 | k2, sizes) * 2;
    double max_size = compute_size_by_dict<T>(k12, sizes);
    return GreedyResult{path, Metrics{log10(flops), log2(max_size), flops, max_size}};
  }

  auto check_outer = check_outer_false<T>;
  if (search_outer) {
    check_outer = check_outer_true;
  }

  CheckContractionType<T> check_contraction = dp_compare_flops<T>;
  if (minimize == Minimize::INTERMEDIATE_SIZE) {
    check_contraction = dp_compare_size<T>;
  }

  const std::vector<double>& size_dict = expression.get_index_to_size();
  std::vector<FrozenSet<T>> inputs;
  inputs.reserve(expression.size() - 1);
  for (uint64_t i = 0; i < expression.size() - 1; ++i) {
    inputs.push_back(expression[i]);
  }
  FrozenSet<T> output = expression[expression.size() - 1];
  std::vector<std::unordered_set<uint64_t>> subgraphs;
  if (search_outer) {
    subgraphs = std::vector<std::unordered_set<uint64_t>>(1);
    for (uint64_t i = 0; i < inputs.size(); ++i) {
      subgraphs[0].insert(i);
    }
  } else {
    subgraphs = find_disconnected_subgraphs(inputs, output);
  }

  std::vector<uint64_t> inputs_contractions;
  inputs_contractions.reserve(inputs.size());
  for (uint64_t i = 0; i < inputs.size(); ++i) {
    inputs_contractions.push_back(i);
  }
  uint64_t all_tensors = (1 << inputs.size()) - 1;
  std::vector<std::shared_ptr<TreeNode<T>>> subgraph_contractions;
  std::vector<double> subgraph_contractions_size;

  uint64_t free_id = inputs.size();

  for (const std::unordered_set<uint64_t>& g : subgraphs) {

    std::vector<std::unordered_map<uint64_t, DPEntry<T>>> x(g.size() + 1);

    for (const auto& j : g) {
      uint64_t key = 1 << j;
      DPEntry<T> value = std::make_tuple(inputs[j], 0.0, 0.0, std::make_shared<TreeNode<T>>(inputs_contractions[j]));
      x[1][key] = value;
    }

    uint64_t g_bitmap = 0;
    for (const auto& j : g) {
      g_bitmap |= (1 << j);
    }

    // indices of the subgraph
    FrozenSet<T> subgraph_inds = bitmap_select<T>(g_bitmap, inputs);

    double cost_cap;
    if (is_cost_cap) {
      auto intersection = subgraph_inds & output;
      cost_cap = compute_size_by_dict(intersection, size_dict);
    } else {
      cost_cap = std::numeric_limits<double>::infinity();
    }
    // compute the cost increment
    double min_size = std::numeric_limits<double>::max();
    for (const auto& ind : subgraph_inds.get_data()) {
      min_size = std::min(min_size, size_dict[ind]);
    }
    double cost_increment = std::max(min_size, 2.0);

    while (x.back().size() == 0) {
      for (size_t n = 2; n <= x[1].size(); ++n) {
        auto& xn = x[n];

        // try to combine solutions from x[m] and x[n-m]
        for (size_t m = 1; m <= n / 2; ++m) {
          for (const auto& [s1, val1] : x[m]) {
            const auto& [i1, cost1, cost1_second, cntrct1] = val1;

            for (const auto& [s2, val2] : x[n - m]) {
              const auto& [i2, cost2, cost2_second, cntrct2] = val2;

              // can only merge if s1 and s2 are disjoint
              // and avoid e.g. s1={0}, s2={1} and s1={1}, s2={0}
              if (!(s1 & s2) && (m != n - m || s1 < s2)) {
                auto i1_cut_i2_wo_output = (i1 & i2) - output;
                // maybe ignore outer products:
                if (check_outer(i1_cut_i2_wo_output)) {
                  auto i1_union_i2 = i1 | i2;
                  check_contraction(cost1, cost2, cost1_second, cost2_second, i1_union_i2, size_dict, cost_cap, s1, s2, xn, g_bitmap, all_tensors, inputs, i1_cut_i2_wo_output, cntrct1, cntrct2);
                }
              }
            }
          }
        }
      }
      // increase cost cap for next iteration:
      cost_cap = cost_increment * cost_cap;
    }
    dynamic_result(x, subgraph_contractions, subgraph_contractions_size, size_dict);
  }

  // create an index array of the same size as subgraph_contractions_size
  std::vector<uint64_t> indices(subgraph_contractions_size.size());
  for (uint64_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  // sort indices based on comparing values from subgraph_contractions_size
  std::sort(indices.begin(), indices.end(), [&subgraph_contractions_size](uint64_t a, uint64_t b) { return subgraph_contractions_size[a] < subgraph_contractions_size[b]; });

  // create a sorted version of subgraph_contractions using the sorted indices
  std::vector<std::shared_ptr<TreeNode<T>>> sorted_contractions;
  for (uint64_t index : indices) {
    sorted_contractions.push_back(subgraph_contractions[index]);
  }
  subgraph_contractions = std::move(sorted_contractions);

  path.ssa.reserve(inputs.size() - 1);
  path.linear.reserve(inputs.size() - 1);
  std::shared_ptr<TreeNode<T>> root = subgraph_contractions[0];
  for (uint64_t i = 1; i < subgraph_contractions.size(); ++i) {
    root = std::make_shared<TreeNode<T>>(std::numeric_limits<T>::max(), root, subgraph_contractions[i]);
  }
  generatePathPairsFromPostorder<T>(root, path.ssa, free_id);
  path.linear = ssa_to_linear(path.ssa);
  auto metrics = compute_metrics(expression, path);
  return GreedyResult{std::move(path), std::move(metrics)};
}
/*********************************************************/

template <class T>
ResultInner switch_to_dynamic(const Expression<T>& expression, Path& path, const std::vector<double>& sizes, const ConsistentOrderedMap<FrozenSet<T>, uint64_t>& remaining, uint64_t free_id, double max_size, double flops, const std::vector<T>& histogram, uint64_t threshold, Minimize minimize, bool is_cost_cap, bool search_outer) {
  std::unordered_map<T, uint64_t> index_to_size_map;
  index_to_size_map.reserve(histogram.size());
  for (uint64_t i = 0; i < histogram.size(); ++i) {
    index_to_size_map.insert({T(i), uint64_t(sizes[i])});
  }
  std::unordered_map<uint64_t, uint64_t> new_to_old_id;
  new_to_old_id.reserve(threshold);
  Expression<T> expression_dp(index_to_size_map);
  uint64_t new_id = 0;
  for (const auto& r : remaining) {
    expression_dp.add(r.first.get_data());
    new_to_old_id[new_id] = r.second;
    new_id++;
  }
  expression_dp.add(expression[expression.size() - 1].get_data());
  expression_dp.simplify();

  auto path_dp = dynamic(expression_dp, minimize, is_cost_cap, search_outer);
  uint64_t offset = free_id - remaining.size();
  for (const auto& pp : path_dp.path.ssa) {
    uint64_t a = pp.a;
    uint64_t b = pp.b;
    if (new_to_old_id.find(a) != new_to_old_id.end()) {
      a = new_to_old_id[a];
    } else {
      a += offset;
    }
    if (new_to_old_id.find(b) != new_to_old_id.end()) {
      b = new_to_old_id[b];
    } else {
      b += offset;
    }
    path.ssa.emplace_back(a, b);
  }
  path.linear = ssa_to_linear(path.ssa);

  return ResultInner{std::move(path), flops + pow(10.0, double(path_dp.metrics.log10_flops)), std::max<double>(max_size, pow(2.0, double(path_dp.metrics.max_log2_size)))};
}

template <class T>
ResultInner greedy_inner(const Expression<T>& expression, uint64_t seed, CostFunctionType<T>& cost_fn, Path path, const std::vector<double>& sizes, const FrozenSet<T>& output, ConsistentOrderedMap<FrozenSet<T>, uint64_t> remaining, uint64_t free_id, ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>> dim_to_keys, double max_size, double flops, std::vector<T> histogram, std::vector<FrozenSet<T>> inputs, uint64_t i_current, const std::atomic<double>& flops_best,
                         const std::atomic<double>& max_size_best, const double ident_size, MinimizeFuncType min_func, bool is_thermal_chooser, const FrozenSet<T>& k_global, const bool generate_linear = true) {

  Minimize minimize = Minimize::INTERMEDIATE_SIZE;
  if(min_func == (MinimizeFuncType) flops_minimize){
    minimize = Minimize::FLOPS;
  }

  bool search_outer = false;
  constexpr uint64_t threshold = 0;
  constexpr bool is_cost_cap = true;
  if (remaining.size() <= threshold && remaining.size() > 2) {
    return switch_to_dynamic(expression, path, sizes, remaining, free_id, max_size, flops, histogram, threshold, minimize, is_cost_cap, search_outer);
  }

  std::priority_queue<CandidateContraction<T>> queue;  // priority queue to store candidate contractions

  const double nTotal = double(remaining.size());

  std::vector<T> k1_free_dims;
  std::vector<T> k2_free_dims;
  std::vector<T> batch_dims;
  std::vector<T> sum_dims;

  uint64_t s0 = (232342352345ull + seed) | 144115188075855872ull;
  uint64_t s1 = (435243623436ull + seed) | 9007199254740992ull;
  // warmup random number generator
  for (int i = 0; i < 5; ++i) {
    next_rand(s0, s1);
  }
  const double alpha = next_double_rand(s0, s1);
  ConsistentOrderedSet<FrozenSet<T>> k2s;
  for (const auto& entry : dim_to_keys) {
    const ConsistentOrderedSet<FrozenSet<T>>& keys = entry.second;
    std::vector<FrozenSet<T>> keys_sorted(keys.begin(), keys.end());
    std::sort(keys_sorted.begin(), keys_sorted.end(), [&](const FrozenSet<T>& a, const FrozenSet<T>& b) { return remaining[a] < remaining[b]; });

    for (uint64_t i = 0; i < keys_sorted.size() - 1; ++i) {
      const FrozenSet<T>& k1 = keys_sorted[i];

      k2s.clear();
      for (uint64_t j = i + 1; j < keys_sorted.size(); ++j) {
        k2s.insert(keys_sorted[j]);
      }

      // call push_candidate to add candidates to the priority queue
      push_candidate<T>(output, sizes, remaining, k1, k2s, queue, cost_fn, s0, s1, nTotal, dim_to_keys, histogram, k1_free_dims, k2_free_dims, batch_dims, sum_dims, ident_size, k_global, alpha);
    }
  }

  // greedily contract pairs of tensors
  while (!queue.empty()) {
    std::optional<std::tuple<double, FrozenSet<T>, FrozenSet<T>, FrozenSet<T>>> result;
    if (is_thermal_chooser) {
      result = thermal_chooser(queue, remaining, s0, s1, (next_rand(s0, s1) % 32) + 1, pow(remaining.size() / double(nTotal), 4 + next_rand(s0, s1) % 8), true);
    } else {
      result = simple_chooser(queue, remaining);
    }
    if (!result) {
      continue;  // allow choose_fn to flag all candidates obsolete
    }

    double cost;
    FrozenSet<T> k1, k2, k12;
    std::tie(cost, k1, k2, k12) = *result;

    uint64_t ssa_id1 = remaining[k1];
    uint64_t ssa_id2 = remaining[k2];
    remaining.erase(k1);
    remaining.erase(k2);

    auto tmp_fset = k1 - output;
    for (const T& dim : tmp_fset.get_data()) {
      dim_to_keys[dim].erase(k1);
    }
    tmp_fset = k2 - output;
    for (const T& dim : tmp_fset.get_data()) {
      dim_to_keys[dim].erase(k2);
    }
    path.ssa.emplace_back(ssa_id1, ssa_id2);
    update_metrics(sizes, histogram, inputs, i_current, ssa_id1, ssa_id2, max_size, flops);
    if (min_func(flops_best, flops, max_size_best, max_size)) {
      return ResultInner{Path{}, std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }

    if (remaining.find(k12) != remaining.end()) {
      path.ssa.push_back(Path_Pair(remaining[k12], free_id++));
      update_metrics(sizes, histogram, inputs, i_current, remaining[k12], free_id - 1, max_size, flops);
      if (min_func(flops_best, flops, max_size_best, max_size)) {
        return ResultInner{Path{}, std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
      }
    } else {
      tmp_fset = k12 - output;
      for (const T& dim : tmp_fset.get_data()) {
        dim_to_keys[dim].insert(k12);
      }
    }
    remaining[k12] = free_id++;

    if (remaining.size() <= threshold && remaining.size() > 2) {
      return switch_to_dynamic(expression, path, sizes, remaining, free_id, max_size, flops, histogram, threshold, minimize, is_cost_cap, search_outer);
    }

    // find new candidate contractions.
    k2s.clear();
    for (const T& dim : k12.get_data()) {
      for (const auto& k2_dim : dim_to_keys[dim]) {
        if (k2_dim != k12) {
          k2s.insert(k2_dim);
        }
      }
    }
    push_candidate<T>(output, sizes, remaining, k12, k2s, queue, cost_fn, s0, s1, nTotal, dim_to_keys, histogram, k1_free_dims, k2_free_dims, batch_dims, sum_dims, ident_size, k_global, alpha);
  }

  // greedily compute pairwise outer products
  if (remaining.size() >= 2) {
    ConsistentOrderedMap<FrozenSet<T>, uint64_t> remaining_sizes;
    for (const auto& entry : remaining) {
      const FrozenSet<T>& key = entry.first & output;
      queue.push(CandidateContraction<T>{compute_size_by_dict(key, sizes) + (next_double_rand(s0, s1) - 0.5) * 0.1, Path_Pair{entry.second, entry.second}, key, FrozenSet<T>{}, FrozenSet<T>{}});
    }
    while (queue.size() >= 2) {
      CandidateContraction<T> c1 = queue.top();
      queue.pop();
      CandidateContraction<T> c2 = queue.top();
      queue.pop();

      FrozenSet<T> k12 = (c1.k1 | c2.k1);
      path.ssa.emplace_back(Path_Pair{c1.pp.a, c2.pp.a});
      queue.push(CandidateContraction<T>{compute_size_by_dict(k12, sizes) + (next_double_rand(s0, s1) - 0.5) * 0.1, Path_Pair{free_id, free_id}, k12, FrozenSet<T>{}, FrozenSet<T>{}});
      update_metrics(sizes, histogram, inputs, i_current, c1.pp.a, c2.pp.a, max_size, flops);
      if (min_func(flops_best, flops, max_size_best, max_size)) {
        return ResultInner{Path{}, std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
      }
      free_id++;
    }
  }

  if(generate_linear) {
    path.linear = ssa_to_linear(path.ssa);
  }

  return ResultInner{std::move(path), flops, max_size};
}

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static int getPhysicalCoreCount() {
  DWORD len = 0;
  GetLogicalProcessorInformation(nullptr, &len);
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  GetLogicalProcessorInformation(buffer.data(), &len);

  int coreCount = 0;
  for (const auto& info : buffer) {
    if (info.Relationship == RelationProcessorCore) {
      coreCount++;
    }
  }
  if(coreCount < 1){
      return int(std::thread::hardware_concurrency());
  }
  return coreCount;
}
#elif defined(__linux__)
static int getPhysicalCoreCount() {
    char buffer[128];
    const char* cmd = "lscpu -p=core,socket | grep -v '#' | sort -u | wc -l";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        return int(std::thread::hardware_concurrency());
    }

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        pclose(pipe);

        int coreCount = std::atoi(buffer);
        if(coreCount < 1){
            return int(std::thread::hardware_concurrency());
        }
        return coreCount;
    }

    pclose(pipe);
    return int(std::thread::hardware_concurrency());
}
#elif defined(__APPLE__)
#include <sys/sysctl.h>
static int getPhysicalCoreCount() {
  int coreCount;
  size_t len = sizeof(coreCount);
  sysctlbyname("hw.physicalcpu", &coreCount, &len, nullptr, 0);

  if(coreCount < 1){
    return int(std::thread::hardware_concurrency());
  }
  return coreCount;
}
#else
static int getPhysicalCoreCount() {
  // Fallback: Return the number of concurrent threads supported by the system.
  // This might include hyperthreads and might not represent the actual physical core count.
  return int(std::thread::hardware_concurrency());
}
#endif

const unsigned int PHYSICAL_CORE_COUNT = (unsigned int)getPhysicalCoreCount();

class TaskQueue {
public:
  TaskQueue(const unsigned int num_threads = PHYSICAL_CORE_COUNT) : num_threads(num_threads == 0 || num_threads > std::thread::hardware_concurrency() ? PHYSICAL_CORE_COUNT : num_threads), stop(false) {}

  // function to add tasks (lambda functions) to the queue
  template <typename Func, typename... Args>
  void addTask(Func&& task, Args&&... args) {
    std::function<void()> boundTask = std::bind(std::forward<Func>(task), std::forward<Args>(args)...);

    {
      std::lock_guard<std::mutex> lock(mtx);
      taskQueue.push(boundTask);
    }
    cv.notify_one();  // notify one waiting thread that a task is available
  }

  // start worker threads to process tasks in parallel
  void startWorkers() {
    //    for (unsigned int i = 0; i < std::thread::hardware_concurrency(); ++i) {
    for (unsigned int i = 0; i < num_threads; ++i) {
      workers.emplace_back(&TaskQueue::processTasks, this);
    }
  }

  // stop worker threads and wait for them to finish
  void stopWorkers() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      stop = true;
    }
    cv.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

private:
  std::queue<std::function<void()>> taskQueue;
  std::vector<std::thread> workers;
  std::mutex mtx;
  std::condition_variable cv;
  const unsigned int num_threads;
  bool stop;

  // function to process tasks in parallel
  void processTasks() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !taskQueue.empty() || stop; });

        if (stop && taskQueue.empty()) {
          return;
        }

        task = std::move(taskQueue.front());
        taskQueue.pop();
      }

      // execute the task
      task();
    }
  }
};

#if defined(_WIN32) || defined(_WIN64)
static void ClearCurrentLineInConsole() {
    HANDLE hConsole = GetStdHandle(STD_ERROR_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    COORD cursorPos;

    if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
        return;

    cursorPos = csbi.dwCursorPosition; // Store the current cursor position
    cursorPos.X = 0; // Go to the start of the line

    DWORD length = csbi.dwSize.X; // Length of the console buffer width
    DWORD written;

    // Fill the entire line with spaces
    FillConsoleOutputCharacter(hConsole, L' ', length, cursorPos, &written);

    // Restore the cursor to the start of the line
    SetConsoleCursorPosition(hConsole, cursorPos);
}
#endif

static void displayProgressBar(uint64_t progress, uint64_t total, double log2Size, double log10Flops) {
  const int barWidth = 20;

  char progressBar[256];
  sprintf(progressBar, "log2[SIZE]: %.2f log10[FLOPs]: %.2f: %.0f%%|", log2Size, log10Flops,
          double(progress) / static_cast<double>(total) * 100.0);

  for (uint64_t i = 0; i < barWidth; ++i) {
    if (i < uint64_t(double(barWidth * progress) / static_cast<double>(total))){
#if defined(_WIN32) || defined(_WIN64)
      strcat(progressBar, "#");
#else
      strcat(progressBar, "█");
#endif
    }
    else
      strcat(progressBar, " ");
  }

  char progressStatus[256];
  sprintf(progressStatus, "| %llu/%llu", progress, total);
  strcat(progressBar, progressStatus);

  // clear the entire line and move the cursor to the beginning
#if defined(_WIN32) || defined(_WIN64)
  ClearCurrentLineInConsole();
#else
  fprintf(stderr, "\033[2K\r");
#endif

  fprintf(stderr, "%s", progressBar);
  fflush(stderr);
}

static inline double checkIdentical(const std::vector<double>& vec) {
  if (vec.empty()) {
    return -1.0;
  }
  double value = vec[0];
  for (double item : vec) {
    if (item != value) {
      return -1.0;  // return -1.0 if elements are not identical
    }
  }
  return value;  // return the identical value
}

class Timer {
private:
  std::chrono::steady_clock::time_point start_time;
  std::chrono::duration<double> duration;
  std::atomic<bool> alwaysFalse;

public:
  // constructor initializes the timer with a given duration in seconds
  explicit Timer(double seconds) : duration(seconds) {
    start_time = std::chrono::steady_clock::now();
    alwaysFalse.store(seconds <= 0);
  }

  // Move constructor
  Timer(Timer&& other) noexcept
      : start_time(std::move(other.start_time)),
        duration(std::move(other.duration)),
        alwaysFalse(other.alwaysFalse.load()) {
    other.alwaysFalse.store(true);
  }

  // Move assignment operator
  Timer& operator=(Timer&& other) noexcept {
    if (this != &other) {
      start_time = std::move(other.start_time);
      duration = std::move(other.duration);
      alwaysFalse.store(other.alwaysFalse.load());
      other.alwaysFalse.store(true);
    }
    return *this;
  }

  // checks if the timer has run out of time
  bool hasTimedOut() const {
    if (alwaysFalse.load()) return false;

    auto current_time = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(current_time - start_time) >= duration;
  }
};


template <class T>
GreedyResult greedy(const Expression<T>& expression, uint64_t seed, uint64_t max_repeats, std::vector<CostFunctionType<T>>& cost_functions, double max_time, bool progbar, Minimize minimize, bool is_outer_optimal, uint32_t threshold_optimal, const unsigned int num_threads, const bool generate_linear = true) {
  MinimizeFuncType min_func = size_minimize;
  if (minimize == Minimize::FLOPS) {
    min_func = flops_minimize;
  }
  Timer timer(max_time);

  Path path;
  const auto sizes = expression.get_index_to_size();
  if (expression.size() < 2) {
    throw std::runtime_error("ERROR: Expression contains less than two tensors.");
  } else if (expression.size() == 2) {
    path.ssa.emplace_back(0, 0);
    path.linear.emplace_back(0, 0);
    double _size = compute_size_by_dict<T>(expression[0], sizes);
    return GreedyResult{path, Metrics{log10(_size * 2), log2(_size), _size * 2, _size}};
  } else if (expression.size() == 3) {
    path.ssa.emplace_back(0, 1);
    path.linear.emplace_back(0, 1);
    const auto& k1 = expression[0];
    const auto& k2 = expression[1];
    const auto& k12 = expression[2];
    double flops = compute_size_by_dict<T>(k1 | k2, sizes) * 2;
    double max_size = compute_size_by_dict<T>(k12, sizes);
    return GreedyResult{path, Metrics{log10(flops), log2(max_size), flops, max_size}};
  }

  if(expression.size() - 1 <= threshold_optimal){
    if (progbar) displayProgressBar(0, 1, log2(std::numeric_limits<double>::max()), log10(std::numeric_limits<double>::max()));
    auto path_dp = dynamic(expression, minimize, true, is_outer_optimal);
    if (progbar) displayProgressBar(1, 1, path_dp.metrics.max_log2_size, path_dp.metrics.log10_flops);
    if (progbar) fprintf(stderr, "\n");
    if (progbar && minimize == Minimize::INTERMEDIATE_SIZE) fprintf(stderr, "optimal size");
    if (progbar && minimize == Minimize::FLOPS) fprintf(stderr, "optimal flops");
    if (progbar){
      if (is_outer_optimal)
        fprintf(stderr, " (search_outer=true)\n");
      else
        fprintf(stderr, " (search_outer=false)\n");
    }
    return path_dp;
  }

  const FrozenSet<T>& output = expression[expression.size() - 1];
  ConsistentOrderedMap<FrozenSet<T>, uint64_t> remaining;
  path.ssa.reserve((expression.size() - 1) + (expression.size() - 2));
  uint64_t free_id = expression.size() - 1;
  // eagerly computing Hadamard products
  for (uint64_t id = 0; id < expression.size() - 1; ++id) {
    const FrozenSet<T>& s = expression[id];
    if (remaining.find(s) == remaining.end()) {
      remaining[s] = id;
    } else {
      path.ssa.push_back({remaining[s], id});
      remaining[s] = free_id++;
    }
  }

  // keep track of possible contraction indices
  ConsistentOrderedMap<T, ConsistentOrderedSet<FrozenSet<T>>> dim_to_keys;
  for (const auto& entry : remaining) {
    FrozenSet<T> key = entry.first;
    FrozenSet<T> simple_key = key - output;
    for (T index : simple_key.get_data()) {
      dim_to_keys[index].insert(key);
    }
  }

  // remove dims that where all tensors are the same
  ConsistentOrderedSet<FrozenSet<uint64_t>> non_duplicate_tensors;
  std::vector<T> dims_to_remove;
  for (const auto& entry : dim_to_keys) {
    T dim = entry.first;
    const ConsistentOrderedSet<FrozenSet<T>>& keySet = entry.second;
    std::vector<uint64_t> tensor_ids;
    for (const auto& key : keySet) {
      tensor_ids.push_back(remaining[key]);
    }
    FrozenSet<uint64_t> s = tensor_ids;
    if (non_duplicate_tensors.find(s) == non_duplicate_tensors.end()) {
      non_duplicate_tensors.insert(s);
    } else {
      dims_to_remove.push_back(dim);
    }
  }
  for (const T& dim : dims_to_remove) {
    dim_to_keys.erase(dim);
  }

  double ident_size = checkIdentical(sizes);

  double max_size = 0;
  double flops = 0;
  std::vector<T> histogram = expression.get_histogram();
  std::vector<FrozenSet<T>> inputs(expression.size() + expression.size() - 1);
  for (uint64_t i = 0; i < expression.size() - 1; ++i) {
    inputs[i] = expression[i];
  }
  uint64_t i = expression.size() - 1;
  for (const auto& contraction : path.ssa) {
    update_metrics(sizes, histogram, inputs, i, contraction.a, contraction.b, max_size, flops);
  }

  uint64_t s0 = (232342352345ull + seed) | 18446744073709551557ull;
  uint64_t s1 = (538008512387ull + seed) | 13493690561280548289ull;
  if (s0 == 0) {
    s0 = 18446744073709551557ull;
  }
  if (s1 == 0) {
    s1 = 13493690561280548289ull;
  }
  // warmup random number generator
  for (int j = 0; j < 5; ++j) {
    next_rand(s0, s1);
  }

  std::atomic<double> flops_best = std::numeric_limits<double>::max();
  std::atomic<double> max_size_best = std::numeric_limits<double>::max();
  Path path_best;
  std::mutex m;
  std::atomic<uint64_t> best_candidate_cost_function = 0;
  std::atomic<uint64_t> current_count = 0;
  // each cost function is evaluated at least 8 times, and then only the best cost function so far is used
  const uint64_t calls_per_cost_function = std::max(uint64_t(8), uint64_t(0.25 * double(max_repeats) / double(cost_functions.size())));
  constexpr uint64_t padding = 256 / sizeof(uint64_t);
  std::vector<std::atomic<uint64_t>> number_of_cost_function_calls(cost_functions.size() * padding);

  std::unordered_set<CostFunctionType<T>> thermal_cost_functions;
  for (uint64_t bi = 0; bi < cost_functions.size(); ++bi) {
    if ((CostFunctionType<T>) cost_functions[bi] == (CostFunctionType<T>) cost_memory_removed_jitter<T>) {
      thermal_cost_functions.insert((CostFunctionType<T>) cost_memory_removed_jitter<T>);
    } else if ((CostFunctionType<T>) cost_functions[bi] == (CostFunctionType<T>) cost_boltzmann<T>) {
      thermal_cost_functions.insert((CostFunctionType<T>) cost_boltzmann<T>);
    } else if ((CostFunctionType<T>) cost_functions[bi] == (CostFunctionType<T>) cost_skew_balanced<T>) {
      thermal_cost_functions.insert((CostFunctionType<T>) cost_skew_balanced<T>);
    } else if ((CostFunctionType<T>) cost_functions[bi] == (CostFunctionType<T>) cost_anti_balanced<T>) {
      thermal_cost_functions.insert((CostFunctionType<T>) cost_anti_balanced<T>);
//        } else if ((CostFunctionType<T>) cost_functions[bi] == (CostFunctionType<T>) cost_test<T>) {
//            thermal_cost_functions.insert((CostFunctionType<T>) cost_test<T>);
    }
  }

  bool is_thermal_chooser = false;
  const uint64_t numTasks = max_repeats;  // number of tasks to be executed
  TaskQueue taskQueue(num_threads);
  taskQueue.startWorkers();

  constexpr double TIME_OUT_PRINT = 1.0 / 60.0;
  Timer timer_print(TIME_OUT_PRINT);

  if (progbar) displayProgressBar(current_count, numTasks, log2(max_size_best), log10(flops_best));

  // add tasks (lambda functions) to the task queue
  for (uint64_t j = 0; j < numTasks; ++j) {
    uint64_t _seed = next_rand(s0, s1);
    if (timer.hasTimedOut() && current_count > 0) break;
    taskQueue.addTask([&, j, _seed, is_thermal_chooser]() mutable {
      if (timer.hasTimedOut() && current_count > 0) return;
      uint64_t cost_function_id = j % cost_functions.size();
      const uint64_t threshold = calls_per_cost_function;
      if (number_of_cost_function_calls[padding * cost_function_id] >= threshold) {
        cost_function_id = best_candidate_cost_function;
      }
      if (thermal_cost_functions.find((CostFunctionType<T>)cost_functions[cost_function_id]) != thermal_cost_functions.end() && number_of_cost_function_calls[padding * cost_function_id] >= 4 * threshold) {
        is_thermal_chooser = true;
      }
      ResultInner result = greedy_inner<T>(expression, _seed, cost_functions[cost_function_id], path, sizes, output, remaining, free_id, dim_to_keys, max_size, flops, histogram, inputs, i, flops_best, max_size_best, ident_size, min_func, is_thermal_chooser, output,
                                           false);
      number_of_cost_function_calls[padding * cost_function_id]++;
      m.lock();
      if (min_func(result.flops, flops_best, result.max_size, max_size_best)) {
        flops_best = result.flops;
        max_size_best = result.max_size;
        path_best = std::move(result.path);
        best_candidate_cost_function = cost_function_id;
      }
      current_count++;
      if (progbar) {
        if(timer_print.hasTimedOut()){
          displayProgressBar(current_count, numTasks, log2(max_size_best), log10(flops_best));
          timer_print = Timer(TIME_OUT_PRINT);
        }
      }
      m.unlock();
    });
  }
  // stop the worker threads and wait for them to finish
  taskQueue.stopWorkers();
  if (progbar) {
    displayProgressBar(current_count, numTasks, log2(max_size_best), log10(flops_best));
    fprintf(stderr, "\n");
    fprintf(stderr, "cost function id of best path: %lld\n", best_candidate_cost_function.load());
  }
  if(generate_linear){
    path_best.ssa = ssa_to_post_order_ssa(path_best.ssa);
    path_best.linear = ssa_to_linear(path_best.ssa);
  }
  return GreedyResult{std::move(path_best), Metrics{log10(flops_best), log2(max_size_best), flops_best, max_size_best}};
}

template <class T>
class CostFunctionsContainer {
public:
  static const std::vector<CostFunctionType<T>>& getCostFunctions() {
    static std::vector<CostFunctionType<T>> cost_functions{
        cost_balanced_boltzmann<T>,     // 0
        cost_boltzmann<T>,              // 1
        cost_max_skew<T>,               // 2
        cost_anti_balanced<T>,          // 3
        cost_skew_balanced<T>,          // 4
        cost_log<T>,                    // 5
        cost_memory_removed_jitter<T>,  // 6
        cost_batch_balanced<T>          // 7
    };
//    static std::vector<CostFunctionType<T>> cost_functions{cost_test<T>};
    return cost_functions;
  }

  CostFunctionsContainer() = delete;
  CostFunctionsContainer(const CostFunctionsContainer&) = delete;
  CostFunctionsContainer& operator=(const CostFunctionsContainer&) = delete;
};

template <class T>
GreedyResult greedy_exec(const Expression<T>& expression, uint64_t seed = 0, uint64_t max_repeats = 128, double max_time = 0.0, bool progbar = false, Minimize minimize = Minimize::INTERMEDIATE_SIZE, bool is_outer_optimal= false, uint32_t threshold_optimal= 12, const unsigned int num_threads = 0, const bool generate_linear = true) {
  auto cost_functions = CostFunctionsContainer<T>::getCostFunctions();
  return greedy<T>(expression, seed, max_repeats, cost_functions, max_time, progbar, minimize, is_outer_optimal, threshold_optimal, num_threads, generate_linear);
}
