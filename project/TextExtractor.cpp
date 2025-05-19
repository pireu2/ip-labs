#include "TextExtractor.h"
#include <regex>

TextExtractor::TextExtractor() : tessApi(new tesseract::TessBaseAPI()), initialized(false) {}

TextExtractor::~TextExtractor() {
    if (initialized) {
        tessApi->End();
    }
    delete tessApi;
}

bool TextExtractor::initialize(const std::string& language) {
    if (tessApi->Init(nullptr, language.c_str()) == 0) {
        initialized = true;

        tessApi->SetPageSegMode(tesseract::PSM_AUTO);
        tessApi->SetVariable("preserve_interword_spaces", "1");

        return true;
    }
    return false;
}

std::string TextExtractor::extractText(const cv::Mat& image) const {
    if (!initialized) {
        return "Error: Tesseract not initialized";
    }

    tessApi->SetImage(image.data, image.cols, image.rows,
                       image.channels(), static_cast<int>(image.step));

    const char* outText = tessApi->GetUTF8Text();
    std::string result(outText);
    delete[] outText;

    return result;
}



std::map<std::string, std::string> TextExtractor::extractPassportInfo(const cv::Mat& image) const {
    std::map<std::string, std::string> passportInfo;

    const std::string text = extractText(image);

    passportInfo["surname"] = parseField(text, "surname");
    passportInfo["given_names"] = parseField(text, "given names");
    passportInfo["passport_no"] = parseField(text, "passport no");
    passportInfo["nationality"] = parseField(text, "nationality");
    passportInfo["date_of_birth"] = parseField(text, "date of birth");
    passportInfo["gender"] = parseField(text, "sex");
    passportInfo["expiry_date"] = parseField(text, "date of expiry");
    passportInfo["mrz"] = parseMRZ(text);

    return passportInfo;
}

std::string TextExtractor::parseField(const std::string& text, const std::string& fieldName) {
    const std::regex pattern("(?i)" + fieldName + "[^a-zA-Z0-9]*([^\n\r]+)");
    std::smatch matches;

    if (std::regex_search(text, matches, pattern) && matches.size() > 1) {
        return matches[1].str();
    }
    return "";
}

std::string TextExtractor::parseMRZ(const std::string& text) {
    const std::regex mrzPattern("P[A-Z<][A-Z<]{3}[A-Z0-9<]{39}[0-9][A-Z0-9<]{42}");
    std::smatch matches;

    if (std::regex_search(text, matches, mrzPattern)) {
        return matches[0].str();
    }
    return "";
}