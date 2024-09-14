import copy
from collections import deque
from typing import List, Deque
from tqdm import tqdm

from arithmetic_encoder.LanguageModelAE import LanguageModel


class Coder:
    def __init__(
        self,
        probability_model: "LanguageModel",
        state_bits: int = 64,
        reset_model_after_finish: bool = False,
    ):
        self.prob_model = probability_model
        self.decoder_prob_model = copy.deepcopy(self.prob_model)
        self.state_bits = (
            state_bits  # The number of bits we use to represent low and high
        )
        self.max_range = (
            1 << state_bits
        )  # The maximum range during coding which is 2^state_bits -> for 8 state_bits 1000000
        self.top_bit = self.max_range >> 1

        self.state_mask = (
            self.max_range - 1
        )  # this is the maximum value that can be stored in self.state bits -> for 8 state bits: 01111111
        self.encoded_result: List[int] = []
        self.low: int = 0  # 0000...0000
        self.high: int = self.state_mask  # the interval is half-opened [low, high)
        # => the maximum value we can store in high is 0111...111 which is 1 if we imagine infinite trailing 1s

        self.underflow_counter: int = 0
        self.encoded: Deque[int] = deque()
        self.code: int = 0
        self.reset_model = reset_model_after_finish

    def process_symbol(
        self,
        total_frequency: int,
        symbol_frequency_low: int,
        symbol_frequency_high: int,
        is_encoding: bool = True,
    ) -> None:
        """Process a single symbol encoding or decoding."""
        new_range = (self.high + 1) - self.low

        low = self.low
        self.low = low + symbol_frequency_low * new_range // total_frequency
        self.high = low + symbol_frequency_high * new_range // total_frequency - 1
        # Check for shift condition: MSB bits of low and high are the same
        msb_bit = self.max_range >> 1

        while ((self.low ^ self.high) & msb_bit) == 0:
            self._handle_shift_state(is_encoding)

        # Check for underflow condition: second MSB of low is 1 and of high is 0
        while (self.low & (~self.high) & (self.max_range >> 2)) != 0:
            self._handle_underflow_state(is_encoding)

    def _handle_shift_state(self, is_encoding: bool):
        """
        we are checking if MSB of low and high are the same
        if they are the same, they form a common prefix which will never change
        we can shift those bits out (to the left) to free up the state space
        shifting the common prefix out is important to maintain precision (we need available state bits)
        """
        if is_encoding:
            low_msb = self.low >> (self.state_bits - 1)  # get the MSB bit
            self.encoded_result.append(low_msb)  # push the MSB bit into output
            # push underflow_counter copies of the opposite bit
            self.encoded_result.extend([low_msb ^ 1] * self.underflow_counter)
            self.underflow_counter = 0
        else:
            encoded_bit = self.encoded.popleft() if len(self.encoded) else 0
            self.code = ((self.code << 1) & self.state_mask) | encoded_bit
        # new low gets shifted 0 to the end
        #  new high gets shifted 1 to the end to keep it the maximum possible below the true high (high + 1)
        self.low = (self.low << 1) & self.state_mask
        self.high = ((self.high << 1) & self.state_mask) | 1

    def _handle_underflow_state(self, is_encoding: bool):
        """
        This function handles the situation when the most significant bit (MSB) of 'low' is 0 and the MSB of 'high' is 1,
        while the second bit from the left of 'low' is 1 and the second bit of 'high' is 0.
        This condition indicates that 'low' and 'high' are getting too close to each other
        in the middle of the range, which may lead to arithmetic underflow.

        Example:
            If low  = 0.0111110101 and
               high = 0.1000001011
            We can remove 5 bits as follows:

            low:  0111110101
                  &
            ~high 0111110100
                = 0111110100
                  &
                  0100000000
                = 0100000000 != 0

        For more information, see:
        https://encode.su/threads/1621-Arithmetic-Coding-Underflow-problem
        """
        if is_encoding:
            self.underflow_counter += 1
        else:
            encoded_bit = self.encoded.popleft() if len(self.encoded) else 0
            old_code = self.code
            self.code &= self.max_range >> 1
            self.code |= ((old_code << 1) & (self.state_mask >> 1)) | encoded_bit

        self.low = (
            self.low << 1
        ) ^ self.top_bit  # shift left and set first bit of low to 0
        # flip top bit to 0, move 1 to top, flip top 1 to zero, shift in 1 from right
        self.high = ((self.high ^ self.top_bit) << 1) | self.top_bit | 1

    def encode(self, data, static=False) -> List[int]:
        for i, symbol in tqdm(enumerate(data), desc="Compressing..."):
            total_freq = self.prob_model.get_total(i)
            symbol_freq_low = self.prob_model.get_low(symbol, i)
            symbol_freq_high = self.prob_model.get_high(symbol, i)

            self.process_symbol(total_freq, symbol_freq_low, symbol_freq_high)
            if not static:
                ...
                # self.prob_model.update_table(symbol)
        self.encoded_result.append(1)
        if self.reset_model:
            self.prob_model.reset_state()
            self.low = 0
            self.high = self.state_mask
            self.underflow_counter: int = 0
        return self.encoded_result

    def decode(self, encoded: List[int], encoded_len: int, static: bool = False):
        self.low: int = 0
        self.high: int = self.state_mask
        self.encoded = deque(encoded)

        for _ in range(self.state_bits):
            encoded_bit = self.encoded.popleft() if len(self.encoded) else 0
            self.code = self.code << 1 | encoded_bit

        result = []
        for i in tqdm(range(encoded_len), desc="Decompressing..."):
            symbol = self._decode_symbol(i)

            if not static:
                self.decoder_prob_model.update_table(symbol)
            result.append(symbol)

        if self.reset_model:
            self.decoder_prob_model.reset_state()
            self.low = 0
            self.high = self.state_mask
            self.underflow_counter: int = 0
            self.encoded: Deque[int] = deque()
            self.code: int = 0
        return result

    def _decode_symbol(self, decoded_symbol_index: int) -> int:
        total_freq = self.decoder_prob_model.get_total(decoded_symbol_index)
        current_range = self.high - self.low + 1
        target = (((self.code - self.low) + 1) * total_freq - 1) // current_range

        #  Find the symbol with the largest low smaller than target
        next_symbol = self.decoder_prob_model.find_correct_interval(
            target, decoded_symbol_index
        )

        symbol_low = self.decoder_prob_model.get_low(next_symbol, decoded_symbol_index)
        symbol_high = self.decoder_prob_model.get_high(
            next_symbol, decoded_symbol_index
        )
        self.process_symbol(total_freq, symbol_low, symbol_high, False)

        return next_symbol


# if __name__ == "__main__":
#     input_data = "In the TV show Homeland, Max Piotrowski, played by Maury Sterling, meets a tragic end in Season 8, Episode 8 titled Threnody(s). Max, a loyal and invaluable member of Carrie Mathisons team, is captured by Taliban forces during a mission in Afghanistan. Despite Carrie's efforts to negotiate his release, Max is ultimately executed by Jalal Haqqani, the new leader of the Taliban, as a demonstration of power and a message to the United States. Max's death is a significant and emotional moment in the series, highlighting the perilous nature of their work and the personal sacrifices made by the characters."
#     freq_table = SymbolFrequencyTable(list(set(input_data)))
#     coder = Coder(freq_table)
#
#     res = coder.encode(input_data, False)
#     print("".join(map(str, res)))
#     decoded = coder.decode(res, len(input_data), False)
#     print("".join(decoded))
#     assert input_data == "".join(decoded)
