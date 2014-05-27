#ifndef KLOSE_CCOLOR
#define KLOSE_CCOLOR
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <iostream>

#define CC_CONSOLE_COLOR_DEFAULT "\033[0m"
#define CC_FORECOLOR(C) "\033[" #C "m"
#define CC_BACKCOLOR(C) "\033[" #C "m"
#define CC_ATTR(A) "\033[" #A "m"

namespace ccolor
{
    enum Color
    {
        Black,
        Red,
        Green,
        Yellow,
        Blue,
        Magenta,
        Cyan,
        White,
        Default = 9
    };

    enum Attributes
    {
        Reset,
        Bright,
        Dim,
        Underline,
        Blink,
        Reverse,
        Hidden
    };

    static inline std::string color(int attr, int fg, int bg)
    {
        char command[13];
        /* Command is the control command to the terminal */
        sprintf(command, "%c[%d;%d;%dm", 0x1B, attr, fg + 30, bg + 40);
        return std::string(command);
    }

    static char const *console = CC_CONSOLE_COLOR_DEFAULT;
    static char const *underline = CC_ATTR(4);
    static char const *bold = CC_ATTR(1);

    namespace fore {
        static char const *black = CC_FORECOLOR(30);
        static char const *blue = CC_FORECOLOR(34);
        static char const *red = CC_FORECOLOR(31);
        static char const *magenta = CC_FORECOLOR(35);
        static char const *green = CC_FORECOLOR(92);
        static char const *cyan = CC_FORECOLOR(36);
        static char const *yellow = CC_FORECOLOR(33);
        static char const *white = CC_FORECOLOR(37);
        static char const *console = CC_FORECOLOR(39);
        static char const *lightblack = CC_FORECOLOR(90);
        static char const *lightblue = CC_FORECOLOR(94);
        static char const *lightred = CC_FORECOLOR(91);
        static char const *lightmagenta = CC_FORECOLOR(95);
        static char const *lightgreen = CC_FORECOLOR(92);
        static char const *lightcyan = CC_FORECOLOR(96);
        static char const *lightyellow = CC_FORECOLOR(93);
        static char const *lightwhite = CC_FORECOLOR(97);
    }

    namespace back {
        static char const *black = CC_BACKCOLOR(40);
        static char const *blue = CC_BACKCOLOR(44);
        static char const *red = CC_BACKCOLOR(41);
        static char const *magenta = CC_BACKCOLOR(45);
        static char const *green = CC_BACKCOLOR(42);
        static char const *cyan = CC_BACKCOLOR(46);
        static char const *yellow = CC_BACKCOLOR(43);
        static char const *white = CC_BACKCOLOR(47);
        static char const *console = CC_BACKCOLOR(49);

        static char const *lightblack = CC_BACKCOLOR(100);
        static char const *lightblue = CC_BACKCOLOR(104);
        static char const *lightred = CC_BACKCOLOR(101);
        static char const *lightmagenta = CC_BACKCOLOR(105);
        static char const *lightgreen = CC_BACKCOLOR(102);
        static char const *lightcyan = CC_BACKCOLOR(106);
        static char const *lightyellow = CC_BACKCOLOR(103);
        static char const *lightwhite = CC_BACKCOLOR(107);
    }

    class Filter {
        std::ostream &os;
        bool isatty;
    public:
        Filter (std::ostream &os_): os(os_) {
            if (&os == &std::cout) {
                isatty = ::isatty(1);
            }
            else if (&os == &std::cerr) {
                isatty = ::isatty(2);
            }
            else {
                isatty = false;
            }
        }
        ~Filter () {
            if (isatty) {
                os << console;
            }
        }
        Filter &operator () (std::string const &stat) {
            os << stat;
            return *this;
        }
    };
}

#endif
