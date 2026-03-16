#include <vector>
#include "types.hpp"
#include "sub/feature.hpp"

// Macro constant
#define VERSION 3

/* Multi-line
macro */
#define SQUARE(x) \
    ((x) * (x))

typedef int Size;

namespace outer {
namespace inner {

// Template box
template <typename T>
class Box {
    using HiddenType = T;
    T hidden_value;
public:
    using ValueType = T;
    Box();
    T get() const;
private:
    T value;
protected:
    void touch() const;
public:
    void set(T value) {
        this->value = value;
    }
};

using IntBox = Box<int>;

enum Color {
    RED,
    GREEN = 2,
    BLUE,
};

union Value {
    int i;
    float f;
};

// Points in 2D space
struct Point {
    int x;
    int y;
};

int sum(int a, int b);

// Add two values
template <typename T>
T add(T a, T b) {
    return a + b;
}

#ifdef ENABLE_FEATURE
int enabled(void);
#else
int disabled(void);
#endif

}

namespace detail::impl {
int helper();
}
}