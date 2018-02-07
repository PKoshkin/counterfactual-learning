#include "active_learning_strategies.h"


std::string PoolBasedUncertaintySamplingStrategy::name() {
    return "uncertainty sampling";
}


double PoolBasedUncertaintySamplingStrategy::get_score(
        CounterfacturalModel* current_model,
        const std::list<Object>& unlabeled_pool,
        const Pool& labeled_pool,
        const Object& obj) {
    std::vector<double> probas = current_model->predict_proba(obj);

    double gini_score = 1;
    for (auto proba: probas)
        gini_score -= proba * proba;

    return gini_score;
}
