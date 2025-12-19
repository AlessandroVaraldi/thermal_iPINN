#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Simple struct for physical parameters, if you ever want to override them */
typedef struct {
    float R_cond;   /* [K/W] */
    float C_th;     /* [J/K] */
} ipinn_phys_params;

/* Fill with default values exported from the training (IPINN_RCOND / IPINN_CTH) */
void ipinn_get_default_phys_params(ipinn_phys_params *out);

/*
 * Compute u_k(t) (correction term) from the TCN network.
 *
 * Inputs:
 *   P_W[t]       : power [W]
 *   T_bplate[t]  : baseplate temperature [°C]
 *   T_amb[t]     : ambient temperature [°C]
 *   T_len        : number of time steps
 *
 * Output:
 *   u_out[t]     : correction term u_k [same units as power]
 *
 * Returns 0 on success, <0 on error (e.g., allocation failure or invalid args).
 */
int ipinn_tcn_predict_u(
    const float *P_W,
    const float *T_bplate,
    const float *T_amb,
    int T_len,
    float *u_out);

/*
 * Full ODE forward pass using the TCN to get u_k(t) and then integrating:
 *
 *   T_{k+1} = T_k + dt/C_th * (
 *     P_k + u_k
 *     - (T_k - T_bplate_k)/R_cond
 *     - (T_k - T_amb_k)/R_conv
 *   )
 *
 * Inputs:
 *   P_W[t], T_bplate[t], T_amb[t] : signals as above
 *   T_len                         : number of time steps
 *   dt                            : timestep [s] (assumed constant)
 *   R_conv                        : convective resistance [K/W] for this scenario
 *   T_init                        : initial temperature T_case at k=0 [°C]
 *
 * Output:
 *   T_pred[t]                     : predicted case temperature [°C]
 *
 * Returns 0 on success, <0 on error.
 */
int ipinn_tcn_predict_temperature(
    const float *P_W,
    const float *T_bplate,
    const float *T_amb,
    int T_len,
    float dt,
    float R_conv,
    float T_init,
    float *T_pred);

/*
 * Self-test against the golden vectors exported in tcn_export.h.
 *
 * Uses:
 *   GOLDEN_P_W, GOLDEN_T_BPLATE, GOLDEN_T_AMB, GOLDEN_T_MEAS,
 *   GOLDEN_T_PRED, GOLDEN_DT, GOLDEN_LEN, IPINN_RCONV_TEST_USED
 *
 * Returns:
 *   max absolute error between engine output and GOLDEN_T_PRED.
 *
 * Note: requires that tcn_export.h was generated with a golden vector.
 */
float ipinn_tcn_self_test(void);

#ifdef __cplusplus
}
#endif
