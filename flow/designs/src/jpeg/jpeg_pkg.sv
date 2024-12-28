package system_config;
    `define CLOCK_FREQUENCY 55615000000    // Define standard width parameters
    parameter INT_WIDTH = 32;
    parameter ADDR_WIDTH = 8;
    parameter DATA_WIDTH = 128;
    parameter DEFAULT_STATE = IDLE;

    // Enumerate common system states
    typedef enum logic [2:0] {
        IDLE = 3'b000,
        INIT = 3'b001,
        ACTIVE = 3'b010,
        ERROR = 3'b011,
        SHUTDOWN = 3'b100
    } system_state_t;

    // Define a structure for configuration settings
    typedef struct packed {
        logic enable;
        logic [3:0] mode;
        logic [7:0] priority;
        logic interrupt_mask;
    } sys_config_t;

    // Custom type for error handling
    typedef enum logic [1:0] {
        NO_ERROR = 2'b00,
        PARITY_ERROR = 2'b01,
        TIMEOUT_ERROR = 2'b10,
        FATAL_ERROR = 2'b11
    } error_type_t;

endpackage
