#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include "argos.h"

using namespace std;
using namespace boost;
namespace po = boost::program_options; 

using namespace argos;

int main (int argc, char *argv[]) {
    string config_path;
    string init_path;
    string model_path;
    unsigned maxloop;
    unsigned report;
    unsigned snapshot;
    bool check = false;
    bool verbose = false;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path), "")
    ("init", po::value(&init_path), "")
    ("model", po::value(&model_path), "")
    ("maxloop", po::value(&maxloop)->default_value(0), "")
    ("report", po::value(&report)->default_value(100), "")
    ("snapshot", po::value(&snapshot)->default_value(0), "")
    ("verbose,v", "")
    ("predict", "")
    ("check", "")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible);

    po::positional_options_description p;
    p.add("config", 1);
    p.add("model", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || vm.count("config") == 0) {
        cout << "Usage: run [OTHER OPTIONS]... <config> [model]" << endl;
        cout << desc_visible << endl;
        return 0;
    }

    if (vm.count("check")) check = true;
    if (vm.count("verbose")) verbose = true;

    Config config;
    load_config(config_path, &config);

    Model model(config);

    if (vm.count("predict")) {
        BOOST_VERIFY(model_path.size());
        model.load(model_path);
        model.predict();
        return 0;
    }

    if (init_path.size()) {
        model.load(init_path);
    }
    else {
        model.init();
    }

    Plan plan;
    model.prepare(MODE_TRAIN, &plan);
    unsigned loop = 0;
    for (;;) {
        plan.run();
        ++loop;
        if (report && (loop % report == 0)) {
            cerr << loop << ' ' << endl;
            cerr << "\ttrain ";
            model.report(cerr);
            cerr << "\ttest ";
            model.predict(cerr);
        }
        if (snapshot && (loop % snapshot == 0)) {
            if (model_path.size()) {
                model.save(model_path + "." + lexical_cast<string>(loop / snapshot));
            }
        }
        if (loop >= maxloop) break;
    }
    if (model_path.size()) {
        model.save(model_path);
    }

    return 0;
}
