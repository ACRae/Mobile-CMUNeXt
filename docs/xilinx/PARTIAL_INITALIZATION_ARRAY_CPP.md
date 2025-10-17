```cpp
// Define 4D array with max dimensions, partially initialize
static float weights[10][8][5][5] = {
    // Only initialize first few "slices"
    {
        { {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f} },  // First 2 rows of first slice
        { {6.0f, 7.0f} }                        // First row of second slice
    },
    {
        { {8.0f, 9.0f, 10.0f, 11.0f} }         // First row of first slice of second "volume"
    }
    // Rest are automatically zero-initialized
};
```
