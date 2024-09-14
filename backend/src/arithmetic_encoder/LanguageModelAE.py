from typing import List, Dict


class LanguageModel:
    """model to be plugged into AECompressor"""

    def __init__(self, cdfs, symbols: List[int]):
        self.cdfs = cdfs
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.freq: Dict[int, int] = {}
        self.context = []
        self.freq_total: int = 0

    def get_total(self, cdf_index):
        return int(self.cdfs[cdf_index][self.symbols[-1]])

    def get_low(self, symbol, cdf_index):
        if symbol == 0:
            return 0
        return int(self.cdfs[cdf_index][symbol - 1])

    def get_high(self, symbol, cdf_index):
        return int(self.cdfs[cdf_index][symbol])

    def update_table(self, symbol):
        self.freq[symbol] = self.freq.get(symbol, 0) + 1

    def entropy(self) -> float:
        """Computes the Shannon binary entropy of the sequence

        self.__freq is a Dict[int, int] with key being token id and value being the frequency over the input sequence

        Returns
        -------
        float
            Shannon entropy in bits
        """
        # frequencies = np.array(list(self.freq.values()))
        # N = sum(frequencies)
        # probs = frequencies / N
        # non_zero_probs = probs[probs > 0]
        # entropy = -sum(non_zero_probs * (np.log2(non_zero_probs)))
        # return entropy
        raise NotImplementedError

    def reset_state(self):
        self.context = []
        self.freq_total: int = 0

    def find_correct_interval(self, target: int, cdf_index):
        symbols = self.symbols

        left = 0
        right = len(self.symbols) - 1
        result = 0
        while left <= right:
            mid = (left + right) // 2
            low = self.get_low(mid, cdf_index)
            if low <= target:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return symbols[result]
