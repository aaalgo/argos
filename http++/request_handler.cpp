//
// request_handler.cpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <fstream>
#include <sstream>
#include <string>
#include <boost/lexical_cast.hpp>
#include "http++.h"

namespace http {
namespace server {

request_handler::~request_handler () {
    for (auto &v: handlers_) {
        delete v.second;
    }
}

request_handler &request_handler::add (std::string const &regexpr, url_handler *handler) {
    handlers_.push_back(std::make_pair(boost::regex(regexpr), handler));
    return *this;
}

void request_handler::handle_request(const request& req, reply& rep)
{
    for (auto &v: handlers_) {
        if (boost::regex_search(req.uri, v.first)) {
            v.second->handle_request(req, rep);
            return;
        }
    }
}

}
}

