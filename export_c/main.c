#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "engine.h"
#include "tcn_export.h"  // contiene GOLDEN_* e IPINN_RCONV_TEST_USED

int main(void) {
    printf("IPINN TCN inference test\n");
    printf("========================\n\n");

    /* --------------------------------------------------------------
     * 1) Alloca buffer per la sequenza predetta
     * -------------------------------------------------------------- */
    float *T_pred = (float *)malloc((size_t)GOLDEN_LEN * sizeof(float));
    if (!T_pred) {
        printf("Error: cannot allocate T_pred\n");
        return EXIT_FAILURE;
    }

    /* --------------------------------------------------------------
     * 2) Esegui l’inferenza completa sullo scenario golden
     *
     *    Ingresso:
     *      GOLDEN_P_W, GOLDEN_T_BPLATE, GOLDEN_T_AMB
     *    Parametri:
     *      dt  = GOLDEN_DT
     *      Rconv = IPINN_RCONV_TEST_USED (o quello che vuoi tu)
     *      T_init = primo valore misurato (stessa convenzione del training)
     * -------------------------------------------------------------- */
    float T_init = GOLDEN_T_MEAS[0];

    int ret = ipinn_tcn_predict_temperature(
        GOLDEN_P_W,
        GOLDEN_T_BPLATE,
        GOLDEN_T_AMB,
        GOLDEN_LEN,
        GOLDEN_DT,
        IPINN_RCONV_TEST_USED,
        T_init,
        T_pred
    );

    if (ret != 0) {
        printf("ipinn_tcn_predict_temperature() returned error %d\n", ret);
        free(T_pred);
        return EXIT_FAILURE;
    }

    /* --------------------------------------------------------------
     * 3) Confronto T_pred vs GOLDEN_T_PRED su TUTTA la sequenza
     *    Metriche: MAE, RMSE, Max |err|
     * -------------------------------------------------------------- */
    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    double max_abs = 0.0;

    for (int i = 0; i < GOLDEN_LEN; ++i) {
        float diff = T_pred[i] - GOLDEN_T_PRED[i];
        float ad   = fabsf(diff);

        sum_abs += (double)ad;
        sum_sq  += (double)diff * (double)diff;

        if (ad > max_abs) {
            max_abs = ad;
        }
    }

    double mae  = sum_abs / (double)GOLDEN_LEN;
    double rmse = sqrt(sum_sq / (double)GOLDEN_LEN);

    printf("Comparison vs GOLDEN_T_PRED (full sequence, N = %d)\n", GOLDEN_LEN);
    printf("  MAE  = %.6f °C\n", mae);
    printf("  RMSE = %.6f °C\n", rmse);
    printf("  Max  |error| = %.6f °C\n\n", max_abs);

    /* (Opzionale) stampa dei primi campioni per sanity check */
    int n_print = (GOLDEN_LEN < 10) ? GOLDEN_LEN : 10;
    printf("First %d samples (time, T_pred, GOLDEN_T_PRED, error):\n", n_print);
    for (int i = 0; i < n_print; ++i) {
        float diff = T_pred[i] - GOLDEN_T_PRED[i];
        printf("  t = %8.3f s,  T_pred = %8.3f,  GOLDEN = %8.3f,  err = %+8.4f\n",
               GOLDEN_TIME[i], T_pred[i], GOLDEN_T_PRED[i], diff);
    }

    free(T_pred);
    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
