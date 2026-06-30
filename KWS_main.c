#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/reboot.h>

#include <arm_math.h>

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "mel_filterbank.h"
#include "dct_matrix.h"


/* ===================================================== */
/* CONFIG */
/* ===================================================== */

#define SAMPLE_RATE            16000

#define FRAME_LEN              512

#define BLOCK_SIZE             (FRAME_LEN * 2)

#define READ_TIMEOUT           1000

#define MFCC_FEATURES          40

#define TOTAL_FEATURES         40

#define PCM_QUEUE_SIZE         4

#define FEATURE_QUEUE_SIZE     4

/* ===================================================== */
/* THREAD STACKS */
/* ===================================================== */

#define AUDIO_STACK_SIZE       4096

#define DSP_STACK_SIZE         8192

#define UART_STACK_SIZE        4096

/* ===================================================== */
/* THREAD PRIORITIES */
/* ===================================================== */

#define AUDIO_PRIORITY         1

#define DSP_PRIORITY           2

#define UART_PRIORITY          3

#define DEBUG_ENABLE           0

#define TRANSMIT_ENABLE        1

#define DISABLE_NORMALIZE      1

/* ===================================================== */
/* MESSAGE QUEUES */
/* ===================================================== */

struct pcm_frame {

    int16_t samples[FRAME_LEN];
};

struct feature_frame {

    float mfcc[TOTAL_FEATURES];
};

K_MSGQ_DEFINE(
    pcm_msgq,
    sizeof(struct pcm_frame),
    PCM_QUEUE_SIZE,
    4
);

K_MSGQ_DEFINE(
    feature_msgq,
    sizeof(struct feature_frame),
    FEATURE_QUEUE_SIZE,
    4
);

/* ===================================================== */
/* MEMORY SLAB */
/* ===================================================== */

K_MEM_SLAB_DEFINE_STATIC(
    mem_slab,
    BLOCK_SIZE,
    128,
    4
);

/* ===================================================== */
/* DEVICES */
/* ===================================================== */

const struct device *dmic_dev;

const struct device *uart_dev;

/* ===================================================== */
/* FFT */
/* ===================================================== */

arm_rfft_fast_instance_f32 fft;

/* ===================================================== */
/* GLOBAL DSP BUFFERS */
/* ===================================================== */

// static float pcm_float[FRAME_LEN];

// static float fft_input[FRAME_LEN];

// static float fft_output[FRAME_LEN];

// static float mfcc[MFCC_FEATURES];

/* ===================================================== */
/* AUDIO THREAD */
/* ===================================================== */

void audio_thread(void)
{
    
    while (1)
    {
        void *buffer;

        uint32_t size;

        int ret = dmic_read(
            dmic_dev,
            0,
            &buffer,
            &size,
            READ_TIMEOUT
        );

        if (ret < 0)
        {
            printk("DMIC READ ERROR\n");
            continue;
        }

        struct pcm_frame frame;

        memcpy(
            frame.samples,
            buffer,
            BLOCK_SIZE
        );

        k_mem_slab_free(
            &mem_slab,
            buffer
        );

        ret = k_msgq_put(
            &pcm_msgq,
            &frame,
            K_NO_WAIT
        );

        if (ret != 0)
        {
            printk("PCM QUEUE FULL\n");
        }
    }
}

/* ===================================================== */
/* DSP THREAD */
/* ===================================================== */

void dsp_thread(void)
{

    struct pcm_frame pcm;
    struct feature_frame feat;

    static float pcm_float[FRAME_LEN];
    static float fft_input[FRAME_LEN];
    static float fft_output[FRAME_LEN];

    static float power_spec[257];
    static float mel_energy[40];
    static float mfcc[40];

    while (1)
    {
        k_msgq_get(
            &pcm_msgq,
            &pcm,
            K_FOREVER
        );

        /* ----------------------------------------- */
        /* PCM -> FLOAT                              */
        /* ----------------------------------------- */

        for (int i = 0; i < FRAME_LEN; i++)
        {
            pcm_float[i] =
                (float)pcm.samples[i] /
                32768.0f;
        }

        /* ----------------------------------------- */
        /* NORMALIZE                                 */
        /* Similar to librosa.util.normalize()       */
        /* ----------------------------------------- */

#if !DISABLE_NORMALIZE

        float max_val = 0.0f;

        for (int i = 0; i < FRAME_LEN; i++)
        {
            float a = fabsf(
                pcm_float[i]
            );

            if (a > max_val)
            {
                max_val = a;
            }
        }

        if (max_val > 1e-6f)
        {
            for (int i = 0; i < FRAME_LEN; i++)
            {
                pcm_float[i] /= max_val;
            }
        }

#endif

        /* ----------------------------------------- */
        /* HANN WINDOW                               */
        /* ----------------------------------------- */

        for (int i = 0; i < FRAME_LEN; i++)
        {
            float window =
                0.5f *
                (
                    1.0f -
                    arm_cos_f32(
                        (2.0f * PI * i) /
                        (FRAME_LEN - 1)
                    )
                );

            fft_input[i] =
                pcm_float[i] *
                window;
        }

        /* ----------------------------------------- */
        /* FFT                                       */
        /* ----------------------------------------- */

        arm_rfft_fast_f32(
            &fft,
            fft_input,
            fft_output,
            0
        );

        /* ----------------------------------------- */
        /* POWER SPECTRUM                            */
        /* ----------------------------------------- */

        power_spec[0] =
            fft_output[0] *
            fft_output[0];

        for (int k = 1; k < 256; k++)
        {
            float real =
                fft_output[2 * k];

            float imag =
                fft_output[2 * k + 1];

            power_spec[k] =
                real * real +
                imag * imag;
        }

        power_spec[256] =
            fft_output[1] *
            fft_output[1];

#if DEBUG_ENABLE

        printk(
            "POWER=%f %f %f %f %f\n",
            power_spec[0],
            power_spec[1],
            power_spec[2],
            power_spec[3],
            power_spec[4]
        );

#endif

        /* ----------------------------------------- */
        /* MEL FILTERBANK                            */
        /* ----------------------------------------- */

        for (int m = 0; m < 40; m++)
        {
            mel_energy[m] = 0.0f;

            for (int k = 0; k < 257; k++)
            {
                mel_energy[m] +=
                    power_spec[k] *
                    mel_filterbank[m][k];
            }
        }

        /* ----------------------------------------- */
        /* power_to_db(ref=np.max) style             */
        /* ----------------------------------------- */

        float mel_max = 0.0f;

        for (int m = 0; m < 40; m++)
        {
            if (mel_energy[m] > mel_max)
            {
                mel_max = mel_energy[m];
            }
        }

        if (mel_max < 1e-10f)
        {
            mel_max = 1e-10f;
        }

        for (int m = 0; m < 40; m++)
        {
            if (mel_energy[m] < 1e-10f)
            {
                mel_energy[m] = 1e-10f;
            }

            mel_energy[m] =
                10.0f *
                log10f(
                    mel_energy[m] /
                    mel_max
                );
        }

#if DEBUG_ENABLE

        printk(
            "MEL=%f %f %f %f %f\n",
            mel_energy[0],
            mel_energy[1],
            mel_energy[2],
            mel_energy[3],
            mel_energy[4]
        );

#endif

        /* ----------------------------------------- */
        /* DCT                                       */
        /* ----------------------------------------- */

        for (int i = 0; i < 40; i++)
        {
            mfcc[i] = 0.0f;

            for (int j = 0; j < 40; j++)
            {
                mfcc[i] +=
                    mel_energy[j] *
                    dct_matrix[i][j];
            }
        }

#if DEBUG_ENABLE

        printk(
            "MFCC[0:5] = [%f %f %f %f %f]\n",
            mfcc[0],
            mfcc[1],
            mfcc[2],
            mfcc[3],
            mfcc[4]
        );

#endif

        /* ----------------------------------------- */
        /* COPY TO FEATURE FRAME                     */
        /* ----------------------------------------- */

        for (int i = 0; i < 40; i++)
        {
            feat.mfcc[i] =
                mfcc[i];
        }

        /* ----------------------------------------- */
        /* SEND TO UART THREAD                       */
        /* ----------------------------------------- */

        int ret =
            k_msgq_put(
                &feature_msgq,
                &feat,
                K_NO_WAIT
            );

        if (ret != 0)
        {
            printk(
                "FEATURE QUEUE FULL\n"
            );
        }
    }
}

/* ===================================================== */
/* UART THREAD */
/* ===================================================== */

void uart_thread(void)
{
    
    struct feature_frame feat;

    uint8_t start_marker[4] = {
        0xAA,
        0xBB,
        0xCC,
        0xDD
    };

    while (1)
    {
        k_msgq_get(
            &feature_msgq,
            &feat,
            K_FOREVER
        );

#if TRANSMIT_ENABLE

        /* ------------------------------------- */
        /* SEND START MARKER                     */
        /* ------------------------------------- */

        for (int i = 0; i < 4; i++)
        {
            uart_poll_out(
                uart_dev,
                start_marker[i]
            );
        }

        /* ------------------------------------- */
        /* SEND 40 MFCC FLOATS                   */
        /* ------------------------------------- */

        uint8_t *ptr =
            (uint8_t *)feat.mfcc;

        int total_bytes =
            sizeof(feat.mfcc);

        for (int i = 0;
             i < total_bytes;
             i++)
        {
            uart_poll_out(
                uart_dev,
                ptr[i]
            );
        }

#endif
    }
}

/* ===================================================== */
/* THREAD DEFINITIONS */
/* ===================================================== */

K_THREAD_DEFINE(
    audio_tid,
    AUDIO_STACK_SIZE,
    audio_thread,
    NULL,
    NULL,
    NULL,
    AUDIO_PRIORITY,
    0,
    0
);

K_THREAD_DEFINE(
    dsp_tid,
    DSP_STACK_SIZE,
    dsp_thread,
    NULL,
    NULL,
    NULL,
    DSP_PRIORITY,
    0,
    0
);

K_THREAD_DEFINE(
    uart_tid,
    UART_STACK_SIZE,
    uart_thread,
    NULL,
    NULL,
    NULL,
    UART_PRIORITY,
    0,
    0
);


static const uint8_t adv_flags[] = {
    BT_LE_AD_GENERAL |
    BT_LE_AD_NO_BREDR
};

static const struct bt_data ad[] = {
    {
        .type = BT_DATA_FLAGS,
        .data_len = sizeof(adv_flags),
        .data = adv_flags,
    },
};


static void start_ble(void)
{
    int err;

    err = bt_enable(NULL);

    if (err)
    {
        printk("Bluetooth init failed\n");
        return;
    }

    err = bt_le_adv_start(
        BT_LE_ADV_CONN,
        ad,
        ARRAY_SIZE(ad),
        NULL,
        0
    );

    if (err)
    {
        printk("Advertising failed\n");
        return;
    }

    printk("Advertising started\n");
}

/* ===================================================== */
/* MAIN */
/* ===================================================== */

int main(void)
{

    int ret;

    /* --------------------------------------------- */
    /* UART */
    /* --------------------------------------------- */

    uart_dev = DEVICE_DT_GET(
        DT_NODELABEL(uart0)
    );

    if (!device_is_ready(uart_dev)) {

        printk("UART NOT READY\n");

        return -1;
    }



    /* --------------------------------------------- */
    /* DMIC */
    /* --------------------------------------------- */

    dmic_dev = DEVICE_DT_GET(
        DT_NODELABEL(pdm0)
    );

    if (!device_is_ready(dmic_dev)) {

        printk("DMIC NOT READY\n");

        return -1;
    }

    

    /* --------------------------------------------- */
    /* PCM STREAM */
    /* --------------------------------------------- */

    struct pcm_stream_cfg stream = {0};

    stream.pcm_rate = SAMPLE_RATE;

    stream.pcm_width = 16;

    stream.block_size = BLOCK_SIZE;

    stream.mem_slab = &mem_slab;

    /* --------------------------------------------- */
    /* DMIC CONFIG */
    /* --------------------------------------------- */

    struct dmic_cfg cfg = {0};

    cfg.io.min_pdm_clk_freq = 1032000;

    cfg.io.max_pdm_clk_freq = 1280000;

    cfg.io.min_pdm_clk_dc = 40;

    cfg.io.max_pdm_clk_dc = 60;

    cfg.streams = &stream;

    cfg.channel.req_num_streams = 1;

    cfg.channel.req_num_chan = 1;

    cfg.channel.req_chan_map_lo =
        dmic_build_channel_map(
            0,
            0,
            PDM_CHAN_RIGHT
        );

    ret = dmic_configure(
        dmic_dev,
        &cfg
    );

    if (ret < 0) {

        printk("DMIC CONFIG FAILED\n");

        return -1;
    }

    //printk("DMIC CONFIG SUCCESS\n");

    /* --------------------------------------------- */
    /* FFT INIT */
    /* --------------------------------------------- */

    arm_rfft_fast_init_f32(
        &fft,
        FRAME_LEN
    );

    /* --------------------------------------------- */
    /* START MIC */
    /* --------------------------------------------- */

    ret = dmic_trigger(
        dmic_dev,
        DMIC_TRIGGER_START
    );

    if (ret < 0) {

        printk("MIC START FAILED\n");

        return -1;
    }

    //printk("MIC STARTED\n");
    /* BLE START */
     start_ble();

    while (1)
    {
        k_sleep(
            K_SECONDS(1)
        );
    }

    return 0;
}