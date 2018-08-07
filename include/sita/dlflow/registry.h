// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef SITA_DLFLOW_REGISTRY_H_
#define SITA_DLFLOW_REGISTRY_H_


#include "sita/macros.h"
#include <map>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "sita/workspace.h"
#include "operator.h"
#include "sita/proto/sita.pb.h"
namespace sita {


template <typename Dtype>
class Operator;

template <typename Dtype>
class GlobalWorkSpace;

template <typename Dtype>
class OperatorRegistry {
 public:
  typedef boost::shared_ptr<Operator<Dtype> > (*Creator)(const OperatorParameter&, GlobalWorkSpace<Dtype> *);
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void  AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Operator type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a operator using a Parameter.
  static boost::shared_ptr<Operator<Dtype> > CreateOperator(const OperatorParameter& param, GlobalWorkSpace<Dtype>* gws) {
    LOG(INFO) << "Creating Operator " << param.name();
    const std::string& type = param.type();
    CreatorRegistry& registry = Registry();

    CHECK_EQ(registry.count(type), 1) << "Unknown Operator type: " << type
        << " (known types: " << OperatorTypeListString() << ")";
    return registry[type](param, gws);
  }

  static std::vector<std::string> OperatorTypeList() {
    CreatorRegistry& registry = Registry();
    std::vector<std::string> operator_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
        operator_types.push_back(iter->first);
    }
    return operator_types;
  }

 private:
  //  registry should never be instantiated - everything is done with its
  // static variables.
  OperatorRegistry() {}

  static std::string OperatorTypeListString() {
    std::vector<std::string> operator_types = OperatorTypeList();
    std::string operator_types_str;
    for (std::vector<std::string>::iterator iter = operator_types.begin();
         iter != operator_types.end(); ++iter) {
      if (iter != operator_types.begin()) {
        operator_types_str += ", ";
      }
      operator_types_str += *iter;
    }
    return operator_types_str;
  }
};


template <typename Dtype>
class OperatorRegisterer {
 public:
    OperatorRegisterer(const std::string& type,
                  boost::shared_ptr<Operator<Dtype> > (*creator)(const OperatorParameter&, GlobalWorkSpace<Dtype>*)) {
   
    OperatorRegistry<Dtype>::AddCreator(type, creator);
  }
};


#define REGISTER_OPERATOR_CREATOR(type, creator)                                  \
  static OperatorRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static OperatorRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_OPERATOR_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  boost::shared_ptr<Operator<Dtype> > Creator_##type(const OperatorParameter& param,GlobalWorkSpace<Dtype>* gws) \
  {                                                                            \
    return boost::shared_ptr<Operator<Dtype> >(new type<Dtype>(param, gws));           \
  }                                                                            \
  REGISTER_OPERATOR_CREATOR(type, Creator_##type)

}  // namespace sita


#endif    //SITA_STUFF_REGISTRY_H_
