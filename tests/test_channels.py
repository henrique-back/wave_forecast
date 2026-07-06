"""
Tests for the configurable channel_set/aux_set input axes:
- prepare_X (frequency-resolved channel stacking)
- prepare_aux (scalar-per-timestep side-input stacking, e.g. wind)
- process_wind (NDBC sentinel masking + wind_u/wind_v conversion)
- WaveHeightBaselineNN with num_channels=1 (density-only) + num_aux_channels=2 (wind)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.prepare_x import prepare_X
from nn.prepare_aux import prepare_aux
from nn.transformer import WaveHeightBaselineNN
from utils.data_processing import process_wind


FREQS = np.array([0.05, 0.10, 0.15, 0.20], dtype=np.float32)


def _freq_df(num_timesteps, num_freqs=4, offset=0.0):
    idx = pd.date_range("2020-01-01", periods=num_timesteps, freq="h")
    data = np.arange(num_timesteps * num_freqs, dtype=np.float32).reshape(num_timesteps, num_freqs) + offset
    return pd.DataFrame(data, index=idx, columns=FREQS[:num_freqs])


class TestPrepareX:
    def test_single_channel_shape(self):
        density = _freq_df(20)
        X = prepare_X([density], seq_length=5, lead_time=2)
        assert X.shape == (20 - 5 - 2 + 1, 5, 4, 1)

    def test_multi_channel_shape_and_order(self):
        density = _freq_df(20, offset=0.0)
        alpha_1 = _freq_df(20, offset=1000.0)
        X = prepare_X([density, alpha_1], seq_length=5, lead_time=2)
        assert X.shape == (20 - 5 - 2 + 1, 5, 4, 2)
        # channel 0 must be density (get_start_token invariant)
        assert torch.allclose(X[0, :, :, 0], torch.from_numpy(density.values[0:5]))
        assert torch.allclose(X[0, :, :, 1], torch.from_numpy(alpha_1.values[0:5]))


class TestPrepareAux:
    def test_empty_channels_gives_zero_width(self):
        aux = prepare_aux([], num_timesteps=20, seq_length=5, lead_time=2)
        assert aux.shape == (20 - 5 - 2 + 1, 5, 0)

    def test_two_channels_shape_and_values(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="h")
        wind_u = pd.Series(np.arange(20, dtype=np.float32), index=idx)
        wind_v = pd.Series(np.arange(20, dtype=np.float32) * -1, index=idx)
        aux = prepare_aux([wind_u, wind_v], num_timesteps=20, seq_length=5, lead_time=2)
        assert aux.shape == (20 - 5 - 2 + 1, 5, 2)
        assert torch.allclose(aux[0, :, 0], torch.from_numpy(wind_u.values[0:5]))
        assert torch.allclose(aux[0, :, 1], torch.from_numpy(wind_v.values[0:5]))


class TestProcessWind:
    def _write_wind_file(self, tmp_path, rows):
        header = "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
        units = "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi    ft\n"
        lines = [header, units]
        for r in rows:
            lines.append(
                f"2020 01 01 {r['hh']:02d} 00 {r['WDIR']} {r['WSPD']} 1.0 "
                "1.0 5.0 5.0 100 1000 20.0 20.0 15.0 10.0 99.0\n"
            )
        path = tmp_path / "wind.txt"
        path.write_text("".join(lines))
        return path

    def test_sentinel_masking_and_uv_conversion(self, tmp_path):
        rows = [
            {"hh": 0, "WDIR": 0,   "WSPD": 5.0},   # wind blowing toward south: (u, v) = (0, -5)
            {"hh": 1, "WDIR": 90,  "WSPD": 5.0},   # wind blowing toward west:  (u, v) = (-5, 0)
            {"hh": 2, "WDIR": 999, "WSPD": 5.0},   # missing direction -> NaN
            {"hh": 3, "WDIR": 180, "WSPD": 99.0},  # missing speed -> NaN
        ]
        self._write_wind_file(tmp_path, rows)
        wind = process_wind(str(tmp_path))

        assert list(wind.columns) == ["wind_u", "wind_v"]
        assert wind["wind_u"].iloc[0] == pytest.approx(0.0, abs=1e-5)
        assert wind["wind_v"].iloc[0] == pytest.approx(-5.0, abs=1e-5)
        assert wind["wind_u"].iloc[1] == pytest.approx(-5.0, abs=1e-5)
        assert wind["wind_v"].iloc[1] == pytest.approx(0.0, abs=1e-5)
        assert np.isnan(wind["wind_u"].iloc[2]) and np.isnan(wind["wind_v"].iloc[2])
        assert np.isnan(wind["wind_u"].iloc[3]) and np.isnan(wind["wind_v"].iloc[3])


class TestModelWithAux:
    def test_forward_pass_density_only_plus_wind(self):
        num_freqs = 4
        seq_len = 6
        lead_time = 3
        batch = 2

        model = WaveHeightBaselineNN(
            freqs=torch.tensor(FREQS),
            num_freqs=num_freqs,
            target='density',
            num_channels=1,
            num_aux_channels=2,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            embed_dim=8,
        )

        src = torch.randn(batch, seq_len, num_freqs, 1)
        aux = torch.randn(batch, seq_len, 2)
        tgt = torch.randn(batch, lead_time, num_freqs)

        out = model(src, tgt, aux=aux)
        assert out.shape == (batch, lead_time, num_freqs)

        inferred = model.infer(src, torch.tensor(FREQS), lead_time, aux=aux)
        assert inferred.shape == (batch, lead_time, num_freqs)
