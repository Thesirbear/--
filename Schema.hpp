#pragma once

#include "block.hpp"
#include <any>

namespace NeuralSchema {

    using Block::Rows;
    using Block::Cols;
    using Block::Generator;

    template<typename T>
    class SchemaBuilder {
        
        public:
        SchemaBuilder<T>() {}
        SchemaBuilder<T>(std::vector<std::any>& x) : SchemaBuilder(x) {}
        
        public:
        class SchemaLinearLayerBuilder {
            
            public:
                SchemaLinearLayerBuilder(std::vector<std::any>& x) :
                    Schema(x)
                {}

                SchemaLinearLayerBuilder LinearLayer(Rows x, Cols y, Generator rng) {
                    Schema.push_back(Block::Linear_Block<T>(x, y, rng));
                    return SchemaLinearLayerBuilder(Schema);
                }

                SchemaBuilder<T> ActivationLayer(Rows x, Cols y, Generator rng) {
                    Schema.push_back(Block::Linear_Block<T>(x, y, rng));
                    return SchemaBuilder<T>(Schema);
                }
            private:
                std::vector<std::any>& Schema;
        };

        SchemaBuilder<T> LinearLayer(Rows x, Cols y, Generator rng) {
            Schema.push_back(Block::Linear_Block<T>(x, y, rng));
            return SchemaBuilder<T>(Schema);
        }

        std::vector<std::any> Schema;
    };


    template<typename T>
    class Schema {
        public:
        
            Schema(SchemaBuilder<T>& X) {
                Blocks(X.Schema);
            }

            Schema(SchemaBuilder<T> X) {
                Blocks(X.Schema);
            }

            std::vector<std::any> Blocks;
            
    };

}