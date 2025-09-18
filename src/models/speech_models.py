"""
Speech Models - For speech feature extraction and analysis
"""
import librosa
import numpy as np

class SpeechModel:
    def init(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def extract_features(self, audio_segment):
        """Extract audio features for analysis"""
        try:
            rms_energy = np.sqrt(np.mean(audio_segment**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate))
            mfcc = np.mean(librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13), axis=1)

            return {
                "rms_energy": rms_energy,
                "zero_crossing_rate": zero_crossing_rate,
                "spectral_centroid": spectral_centroid,
                "mfcc": mfcc
            }
        except Exception as e:
            print(f"⚠️ Speech feature extraction error: {e}")
            return None
