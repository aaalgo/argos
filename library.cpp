#include <dlfcn.h>
#include "argos.h"

namespace argos {

    void Library::load (string const &path) {
        LOG(info) << "Loading library " << path;
        if (m_modules.find(path) != m_modules.end()) throw runtime_error("module already loaded " + path);
        void *dl = dlopen(path.c_str(), RTLD_NOW);
        if (dl == 0) throw runtime_error("cannot load dl " + path + " " + dlerror());
        void *entry = dlsym(dl, "ArgosRegisterLibrary");
        if (entry == 0) throw runtime_error("cannot find argos entry from " + path + " " + dlerror());
        ((ArgosRegisterLibraryFunctionType)entry)(this);
        auto r = m_modules.insert(make_pair(path, dl));
        BOOST_VERIFY(r.second);
    }

    void Library::cleanupModules () {
        for (auto p: m_modules) {
            dlclose(p.second);
        }
    }
}
