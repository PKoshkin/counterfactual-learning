#include "utils.h"

template<class Model>
class ALStrategy {
public:
    virtual vector<uint16_t> get_batch(const Pool& pool,
                                       const Model& current_model,
                                       const std::set<uint16_t>& unlabeled_indexes,
                                       const Pool& labeled_pool);
private:
    uint16_t batch_size;
};
