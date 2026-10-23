/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <stdint.h>

#include "trace_record.h"

static __managed__ uint64_t total_dynamic_instr_counter = 0;
static __managed__ uint64_t reported_dynamic_instr_counter = 0;
static __managed__ bool stop_report = false;
