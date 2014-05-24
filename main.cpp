#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <boost/log/utility/setup/console.hpp>
#include "argos.h"

using namespace std;
using namespace boost;
namespace po = boost::program_options; 
namespace logging = boost::log;

using namespace argos;

int main (int argc, char *argv[]) {
    string config_path;
    string init_path;
    string model_path;
    unsigned maxloop;
    unsigned report;
    unsigned snapshot;
    string check;
    double epsilon;
    unsigned sample;
    int loglevel; // = logging::trivial::info;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("config", po::value(&config_path), "")
    ("init", po::value(&init_path), "")
    ("model", po::value(&model_path), "")
    ("maxloop", po::value(&maxloop)->default_value(0), "")
    ("report", po::value(&report)->default_value(100), "")
    ("snapshot", po::value(&snapshot)->default_value(0), "")
    ("level", po::value(&loglevel)->default_value(logging::trivial::info), "")
    ("check", po::value(&check), "")
    ("check-epsilon", po::value(&epsilon)->default_value(0.0001), "")
    ("check-sample", po::value(&sample)->default_value(10), "")
    ("predict", "")
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

    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    logging::add_console_log(cerr);
    logging::core::get()->set_filter(logging::trivial::severity >= loglevel);

    Config config;
    LoadConfig(config_path, &config);

    config.put("argos.global.report", report);
    config.put("argos.global.snapshot", snapshot);
    config.put("argos.global.maxloop", maxloop);
    config.put("argos.global.model", model_path);


    if (vm.count("predict")) {
        Model model(config, MODE_PREDICT);
        BOOST_VERIFY(model_path.size());
        model.load(model_path);
        model.predict();
        return 0;
    }
    else if (vm.count("check")) {
        Model model(config, MODE_TRAIN);
        if (init_path.size()) {
            model.load(init_path);
        }
        else {
            model.init();
        }
        model.verify(check, epsilon, sample);
    }
    else {
        Model model(config, MODE_TRAIN);
        if (init_path.size()) {
            model.load(init_path);
        }
        else {
            model.init();
        }
        model.train();
        if (model_path.size()) {
            model.save(model_path);
        }
    }
    return 0;
}
