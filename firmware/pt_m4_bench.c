
#include <stdint.h>
static float pt_buf[32];
uint32_t PanTompkins_M4_Step(float x, uint32_t idx) {
    static float z1[2], z2[2], prev_y, spki = 0.25f, npki = 0.01f;
    static uint32_t last_pk = 0, ptr = 0;
    float w = x + 1.14298f*z1[0] - 0.41280f*z2[0];
    float y = 0.06745f*(w - z2[0]);
    z2[0] = z1[0]; z1[0] = w;
    float d = y - prev_y; prev_y = y;
    float sq = d * d;
    pt_buf[ptr & 31U] = sq; ptr++;
    float mwi = 0.0f;
    for (int k = 0; k < 15; k++) mwi += pt_buf[(ptr - 1U - (uint32_t)k) & 31U];
    mwi *= (1.0f / 15.0f);
    float th = npki + 0.25f * (spki - npki);
    if ((idx - last_pk) >= 25U && mwi > th) {
        spki = 0.125f * mwi + 0.875f * spki; last_pk = idx; return 1U;
    }
    npki = 0.125f * mwi + 0.875f * npki;
    return 0U;
}
int main(void) { return (int)PanTompkins_M4_Step(0.5f, 30U); }
