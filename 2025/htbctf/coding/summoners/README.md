# Solution


    #include <iostream>
    #include <string>
    #include <vector>
    #include <sstream> // Include stringstream
    using namespace std;

    int main() {
        string input_text;
        getline(cin, input_text);

        vector<int> nums;
        stringstream ss(input_text);
        string token;

        // Remove the opening '['
        if (input_text[0] == '[') {
            input_text = input_text.substr(1);
            ss.str(input_text); // Update stringstream
        }

        while (getline(ss, token, ',')) {
            // Remove the closing ']' if it exists
            if (token.back() == ']') {
                token.pop_back();
            }
            nums.push_back(stoi(token));
        }

        int n = nums.size();
        int prev = nums[0];
        int prev2 = 0;

        for (int i = 1; i < n; i++) {
            int pick = nums[i];
            if (i > 1) pick += prev2;
            int notpick = prev;
            int curri = max(pick, notpick);
            prev2 = prev;
            prev = curri;
        }
        cout << prev;
        return 0;
    }

