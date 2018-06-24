#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <iterator>
#include <fstream>


const int N_WORDS = 370752;


typedef std::vector<int> Query;


double get_similarity(const Query& query_1, const Query& query_2) {
    std::vector<int> union_dest;
    std::vector<int> intersection_dest;
 
    std::set_union(query_1.begin(), query_1.end(),
                   query_2.begin(), query_2.end(),
                   std::back_inserter(union_dest));
    std::set_intersection(query_1.begin(), query_1.end(),
                          query_2.begin(), query_2.end(),
                          std::back_inserter(intersection_dest));
    return (static_cast<float>(intersection_dest.end() - intersection_dest.begin()) /
            static_cast<float>(union_dest.end() - union_dest.begin()));
}


void print_query(const Query& query, std::ostream& stream) {
    stream << "(";
    for (int i = 0; i < query.size(); ++i) {
        stream << query[i];
        if (i != query.size() - 1) {
            stream << ", ";
        } else {
            stream << ")";
        }
    }
}


int main() {
    std::fstream input("queries.txt", input.in);

    std::vector<Query> queries;
    for (int i = 0; i < 459862; ++i) {
        int len;
        input >> len;
        Query query;
        for (int j = 0; j < len; ++j) {
            int word;
            input >> word;
            query.push_back(word);
        }
        queries.push_back(query);
    }

    std::vector<std::vector<Query>> queries_by_word;
    for (int i = 0; i < N_WORDS; ++i) {
        queries_by_word.emplace_back();
    }
    for (int i = 0; i < queries.size(); ++i) {
        for (int j = 0; j < queries[i].size(); ++j) {
            queries_by_word[queries[i][j]].push_back(queries[i]);
        }
    }


    //int i = 0;
    //for (int j = 0; j < queries_by_word[i].size(); ++j) {
    //    print_query(queries_by_word[i][j], std::cout);
    //    std::cout << std::endl;
    //}

    std::fstream output("nearest_queries.txt", input.out);
    output << "{" << std::endl;
    for (int i = 0; i < queries.size(); ++i) {
        if (i % 500 == 0) {
            std::cout << i << " / " << queries.size() << " " << i * 100 / queries.size() << " %" << std::endl;
        }
        print_query(queries[i], output);
        output << ": ";
        std::set<Query> candidates;
        for (int word_ind = 0; word_ind < queries[i].size(); ++word_ind) {
            int queries_canditates_number = queries_by_word[queries[i][word_ind]].size();
            for (int other_query_ind = 0; other_query_ind < queries_canditates_number; ++other_query_ind) {
                Query candidate = queries_by_word[queries[i][word_ind]][other_query_ind];
                double similarity = get_similarity(queries[i], candidate);
                if (similarity >= 0.5) {
                    candidates.insert(candidate);
                }
            }
        }
        output << " [";
        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            if (it != candidates.begin()) {
                output << ", ";
            }
            print_query(*it, output);
        }
        output << "]" << std::endl;
    }

    output << "}";
    return 0;
}
