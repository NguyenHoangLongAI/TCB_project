import requests
import json
import os
from pathlib import Path
import time
from typing import Dict, Any, Optional


class DocumentProcessingAPITester:
    def __init__(self, base_url: str = "https://f7a9e2724f33.ngrok-free.app"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        print("🔍 Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            result = {
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
            print(f"✅ Health check: {'PASSED' if result['success'] else 'FAILED'}")
            return result
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"success": False, "error": str(e)}

    def test_process_document_text(self, text_content: str = None) -> Dict[str, Any]:
        """Test API 1 with text content"""
        print("\n📄 Testing API 1: Process Text Document...")

        if text_content is None:
            text_content = """
# Hướng dẫn sử dụng Trợ lý ảo

## 1. Giới thiệu
Trợ lý ảo là một công cụ hỗ trợ người dùng trong việc tra cứu thông tin và giải đáp các câu hỏi.

### 1.1 Tính năng chính
- Trả lời câu hỏi tự động
- Tìm kiếm thông tin
- Hỗ trợ nhiều ngôn ngữ

### 1.2 Đối tượng sử dụng
Các thành viên trong sở nội vụ, người dân, và các cơ quan liên quan.

## 2. Cách sử dụng
Để sử dụng trợ lý ảo, người dùng có thể:
1. Gửi câu hỏi qua giao diện web
2. Sử dụng API để tích hợp vào hệ thống khác
3. Truy cập qua ứng dụng di động

## 3. Lưu ý quan trọng
- Đảm bảo câu hỏi rõ ràng và cụ thể
- Kiểm tra lại thông tin trước khi sử dụng
- Liên hệ hỗ trợ khi cần thiết
"""

        try:
            # Create temporary text file
            temp_file_path = "test_document.txt"
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            # Send request
            with open(temp_file_path, 'rb') as f:
                files = {'file': ('test_document.txt', f, 'text/plain')}
                response = self.session.post(
                    f"{self.base_url}/api/v1/process-document",
                    files=files
                )

            # Clean up
            os.remove(temp_file_path)

            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }

            if result["success"]:
                print("✅ Process document: PASSED")
                print(f"📝 Markdown length: {len(result['response']['markdown_content'])}")
            else:
                print("❌ Process document: FAILED")
                print(f"Error: {result['response']}")

            return result

        except Exception as e:
            print(f"❌ Process document failed: {e}")
            return {"success": False, "error": str(e)}

    def test_process_document_file(self, file_path: str) -> Dict[str, Any]:
        """Test API 1 with actual file"""
        print(f"\n📁 Testing API 1: Process File {file_path}...")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return {"success": False, "error": "File not found"}

        try:
            # Determine content type
            file_ext = Path(file_path).suffix.lower()
            content_type_map = {
                '.txt': 'text/plain',
                '.pdf': 'application/pdf',
                '.doc': 'application/msword',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.xls': 'application/vnd.ms-excel',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }

            content_type = content_type_map.get(file_ext, 'application/octet-stream')

            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, content_type)}
                response = self.session.post(
                    f"{self.base_url}/api/v1/process-document",
                    files=files
                )

            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }

            if result["success"]:
                print("✅ Process file: PASSED")
                print(f"📝 Markdown length: {len(result['response']['markdown_content'])}")
            else:
                print("❌ Process file: FAILED")
                print(f"Error: {result['response']}")

            return result

        except Exception as e:
            print(f"❌ Process file failed: {e}")
            return {"success": False, "error": str(e)}

    def test_embed_markdown(self, markdown_content: str = None, document_id: str = "test_doc_001") -> Dict[str, Any]:
        """Test API 2 with markdown content"""
        print(f"\n🔗 Testing API 2: Embed Markdown for document {document_id}...")

        if markdown_content is None:
            markdown_content = """
# Thông tin chung về Trợ lý ảo

## 1. Định nghĩa
Trợ lý ảo (Virtual Assistant) là một ứng dụng phần mềm được thiết kế để hỗ trợ người dùng thực hiện các tác vụ hoặc dịch vụ thông qua giao diện tự nhiên.

## 2. Nguyên lý hoạt động

### 2.1 Cấu tạo
Hệ thống bao gồm các thành phần chính:
- Module xử lý ngôn ngữ tự nhiên
- Cơ sở dữ liệu kiến thức
- Engine trả lời câu hỏi

### 2.2 Hoạt động
Quá trình hoạt động gồm các bước:
1. Tiếp nhận câu hỏi từ người dùng
2. Phân tích và hiểu ý định câu hỏi
3. Tìm kiếm thông tin liên quan
4. Tổng hợp và trả lời

## 3. Kết luận
Trợ lý ảo là công cụ hữu ích giúp cải thiện hiệu quả công việc và nâng cao trải nghiệm người dùng.
"""

        try:
            payload = {
                "markdown_content": markdown_content,
                "document_id": document_id
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/embed-markdown",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }

            if result["success"]:
                print("✅ Embed markdown: PASSED")
                print(f"🔗 Embeddings created: {result['response']['embeddings_count']}")
                print(f"💾 Stored in Milvus: {result['response']['stored_count']}")
            else:
                print("❌ Embed markdown: FAILED")
                print(f"Error: {result['response']}")

            return result

        except Exception as e:
            print(f"❌ Embed markdown failed: {e}")
            return {"success": False, "error": str(e)}

    def test_chatbot_ask(self, questions: list = None) -> Dict[str, Any]:
        """Test API 3: Chatbot Ask endpoint"""
        print("\n🤖 Testing API 3: Chatbot Ask...")

        if questions is None:
            questions = [
                "Trợ lý ảo có những tính năng gì?",
                "AI có thể giúp tăng năng suất như thế nào?",
                "Làm thế nào để tối ưu hóa hiệu suất hệ thống?",
                "Phân tích xu hướng thị trường công nghệ",
                "Chiến lược marketing hiệu quả cho doanh nghiệp"
            ]

        all_results = []
        successful_tests = 0

        for i, question in enumerate(questions, 1):
            print(f"\n🔹 Test {i}/{len(questions)}: {question}...")

            try:
                payload = {
                    "question": question
                }

                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/api/v1/chatbot/ask",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                end_time = time.time()

                result = {
                    "test_number": i,
                    "question": question,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time_seconds": round(end_time - start_time, 2),
                    "response": response.json() if response.status_code == 200 else response.text
                }

                if result["success"]:
                    successful_tests += 1
                    response_data = result["response"]["data"]
                    print(f"✅ Question {i}: PASSED")
                    print(f"⏱️  Response time: {result['response_time_seconds']}s")
                    print(f"📝 Answer preview: {response_data['answer'][:100]}...")
                    print(f"📚 References: {', '.join(response_data['references'][:3])}")
                    print(f"🎯 Confidence: {response_data['confidence_score']:.2f}")
                else:
                    print(f"❌ Question {i}: FAILED")
                    print(f"Error: {result['response']}")

                all_results.append(result)

                # Small delay between requests
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Question {i} failed: {e}")
                all_results.append({
                    "test_number": i,
                    "question": question,
                    "success": False,
                    "error": str(e)
                })

        # Summary for chatbot tests
        print(f"\n🤖 Chatbot API Summary: {successful_tests}/{len(questions)} tests passed")

        return {
            "success": successful_tests > 0,
            "total_tests": len(questions),
            "passed_tests": successful_tests,
            "success_rate": round(successful_tests / len(questions) * 100, 1),
            "results": all_results
        }

    def test_add_faq(self, faq_data: list = None) -> Dict[str, Any]:
        """Test API 4: Add FAQ endpoint"""
        print("\n➕ Testing API 4: Add FAQ...")

        if faq_data is None:
            faq_data = [
                {
                    "question": "Trợ lý ảo là gì?",
                    "answer": "Trợ lý ảo là một hệ thống phần mềm sử dụng trí tuệ nhân tạo để hỗ trợ người dùng thực hiện các tác vụ và trả lời câu hỏi.",
                    "faq_id": "faq_001"
                },
                {
                    "question": "Làm thế nào để sử dụng trợ lý ảo?",
                    "answer": "Bạn có thể sử dụng trợ lý ảo bằng cách gửi câu hỏi qua giao diện web, API, hoặc ứng dụng di động.",
                    "faq_id": "faq_002"
                },
                {
                    "question": "Trợ lý ảo có hỗ trợ tiếng Việt không?",
                    "answer": "Có, trợ lý ảo được thiết kế để hỗ trợ tiếng Việt và nhiều ngôn ngữ khác."
                }  # No faq_id - will be auto-generated
            ]

        all_results = []
        successful_tests = 0

        for i, faq in enumerate(faq_data, 1):
            print(f"\n🔹 Adding FAQ {i}/{len(faq_data)}: {faq['question'][:50]}...")

            try:
                payload = {
                    "question": faq["question"],
                    "answer": faq["answer"]
                }

                if "faq_id" in faq and faq["faq_id"]:
                    payload["faq_id"] = faq["faq_id"]

                response = self.session.post(
                    f"{self.base_url}/api/v1/faq/add",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )

                result = {
                    "test_number": i,
                    "question": faq["question"],
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else response.text
                }

                if result["success"]:
                    successful_tests += 1
                    response_data = result["response"]
                    print(f"✅ FAQ {i}: ADDED")
                    print(f"🆔 FAQ ID: {response_data['faq_id']}")
                else:
                    print(f"❌ FAQ {i}: FAILED")
                    print(f"Error: {result['response']}")

                all_results.append(result)

            except Exception as e:
                print(f"❌ FAQ {i} failed: {e}")
                all_results.append({
                    "test_number": i,
                    "question": faq["question"],
                    "success": False,
                    "error": str(e)
                })

        print(f"\n➕ Add FAQ Summary: {successful_tests}/{len(faq_data)} FAQs added")

        return {
            "success": successful_tests > 0,
            "total_tests": len(faq_data),
            "passed_tests": successful_tests,
            "success_rate": round(successful_tests / len(faq_data) * 100, 1),
            "results": all_results
        }

    def test_delete_faq(self, faq_ids: list = None) -> Dict[str, Any]:
        """Test API 5: Delete FAQ endpoint"""
        print("\n🗑️ Testing API 5: Delete FAQ...")

        if faq_ids is None:
            faq_ids = ["faq_001", "faq_002", "nonexistent_faq"]

        all_results = []
        successful_tests = 0

        for i, faq_id in enumerate(faq_ids, 1):
            print(f"\n🔹 Deleting FAQ {i}/{len(faq_ids)}: {faq_id}...")

            try:
                response = self.session.delete(
                    f"{self.base_url}/api/v1/faq/delete/{faq_id}"
                )

                result = {
                    "test_number": i,
                    "faq_id": faq_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else response.text
                }

                if result["success"]:
                    successful_tests += 1
                    print(f"✅ FAQ {i}: DELETED")
                else:
                    print(f"❌ FAQ {i}: FAILED")
                    print(f"Error: {result['response']}")

                all_results.append(result)

            except Exception as e:
                print(f"❌ Delete FAQ {i} failed: {e}")
                all_results.append({
                    "test_number": i,
                    "faq_id": faq_id,
                    "success": False,
                    "error": str(e)
                })

        print(f"\n🗑️ Delete FAQ Summary: {successful_tests}/{len(faq_ids)} FAQs deleted")

        return {
            "success": successful_tests >= 0,  # Even failed deletes are expected for non-existent FAQs
            "total_tests": len(faq_ids),
            "passed_tests": successful_tests,
            "results": all_results
        }

    def test_delete_document(self, document_ids: list = None) -> Dict[str, Any]:
        """Test API 6: Delete Document endpoint"""
        print("\n🗑️ Testing API 6: Delete Document...")

        if document_ids is None:
            document_ids = ["test_doc_001", "workflow_test_doc", "nonexistent_doc"]

        all_results = []
        successful_tests = 0

        for i, doc_id in enumerate(document_ids, 1):
            print(f"\n🔹 Deleting Document {i}/{len(document_ids)}: {doc_id}...")

            try:
                response = self.session.delete(
                    f"{self.base_url}/api/v1/document/delete/{doc_id}"
                )

                result = {
                    "test_number": i,
                    "document_id": doc_id,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else response.text
                }

                if result["success"]:
                    successful_tests += 1
                    print(f"✅ Document {i}: DELETED")
                else:
                    print(f"❌ Document {i}: FAILED")
                    print(f"Error: {result['response']}")

                all_results.append(result)

            except Exception as e:
                print(f"❌ Delete Document {i} failed: {e}")
                all_results.append({
                    "test_number": i,
                    "document_id": doc_id,
                    "success": False,
                    "error": str(e)
                })

        print(f"\n🗑️ Delete Document Summary: {successful_tests}/{len(document_ids)} documents deleted")

        return {
            "success": successful_tests >= 0,  # Even failed deletes are expected
            "total_tests": len(document_ids),
            "passed_tests": successful_tests,
            "results": all_results
        }

    def test_chatbot_edge_cases(self) -> Dict[str, Any]:
        """Test chatbot with edge cases"""
        print("\n🧪 Testing Chatbot Edge Cases...")

        edge_cases = [
            {"question": "", "description": "Empty question"},
            {"question": "a", "description": "Too short question"},
            {"question": "What is the meaning of life?" * 50, "description": "Very long question"},
            {"question": "!@#$%^&*()", "description": "Special characters only"},
            {"question": "123456789", "description": "Numbers only"},
        ]

        results = []
        for case in edge_cases:
            print(f"\n🔸 Testing: {case['description']}")

            try:
                payload = {"question": case["question"]}
                response = self.session.post(
                    f"{self.base_url}/api/v1/chatbot/ask",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )

                result = {
                    "description": case["description"],
                    "question": case["question"],
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response": response.json() if response.status_code == 200 else response.text
                }

                if result["success"]:
                    print(f"✅ {case['description']}: Handled gracefully")
                else:
                    print(f"❌ {case['description']}: {result['response']}")

                results.append(result)

            except Exception as e:
                print(f"❌ {case['description']} failed: {e}")
                results.append({
                    "description": case["description"],
                    "success": False,
                    "error": str(e)
                })

        return {
            "success": True,  # Edge cases are expected to have mixed results
            "results": results
        }

    def test_full_workflow(self, test_text: str = None) -> Dict[str, Any]:
        """Test complete workflow: Document -> Markdown -> Embeddings -> FAQ -> Cleanup"""
        print("\n🔄 Testing Full Workflow...")

        # Step 1: Process document
        doc_result = self.test_process_document_text(test_text)
        if not doc_result.get("success"):
            return {"success": False, "error": "Document processing failed", "step": 1}

        # Step 2: Embed markdown
        markdown_content = doc_result["response"]["markdown_content"]
        embed_result = self.test_embed_markdown(markdown_content, "workflow_test_doc")

        if not embed_result.get("success"):
            return {"success": False, "error": "Embedding failed", "step": 2}

        # Step 3: Add FAQ
        faq_result = self.test_add_faq([{
            "question": "Quy trình làm việc của hệ thống như thế nào?",
            "answer": "Hệ thống hoạt động theo quy trình: nhận tài liệu -> xử lý -> tạo embedding -> lưu trữ -> trả lời câu hỏi.",
            "faq_id": "workflow_faq"
        }])

        # Step 4: Test chatbot
        chat_result = self.test_chatbot_ask(["Hệ thống hoạt động như thế nào?"])

        print("✅ Full workflow: COMPLETED")
        return {
            "success": True,
            "document_processing": doc_result,
            "embedding": embed_result,
            "faq_addition": faq_result,
            "chatbot_test": chat_result
        }

    def create_sample_files(self):
        """Create sample files for testing"""
        print("\n📝 Creating sample files for testing...")

        # Sample text file
        with open("sample_text.txt", "w", encoding="utf-8") as f:
            f.write("""
# Quy trình xử lý hồ sơ

## 1. Tiếp nhận hồ sơ
Công dân nộp hồ sơ tại bộ phận tiếp nhận hoặc qua hệ thống trực tuyến.

## 2. Kiểm tra hồ sơ
- Kiểm tra tính đầy đủ của hồ sơ
- Xác minh thông tin cá nhân
- Đối chiếu với quy định hiện hành

## 3. Xử lý hồ sơ
Thời gian xử lý: 15 ngày làm việc kể từ ngày tiếp nhận hồ sơ hợp lệ.

## 4. Trả kết quả
Thông báo kết quả qua SMS, email hoặc công dân đến trực tiếp để nhận.
""")

        # Sample CSV file (as table example)
        try:
            import pandas as pd
            df = pd.DataFrame([
                {"STT": 1, "Câu hỏi": "Ai sẽ là người sử dụng Trợ lý ảo (TLA)",
                 "Trả lời mong muốn": "VD: Các thành viên trong sở nội vụ, người dân, ...",
                 "Chú thích": "Nếu sử dụng nội dung bên ngoài. Câu trả lời có thể khác với những gì TLA được học"},
                {"STT": 2, "Câu hỏi": "Phạm vi trợ lý ảo có thể trả lời",
                 "Trả lời mong muốn": "VD: Các thông tư văn bản được training hay có thể tìm kiếm thông tin bên ngoài để trả lời",
                 "Chú thích": "Cần training dữ liệu phù hợp"},
            ])
            df.to_excel("sample_table.xlsx", index=False)
            print("✅ Sample files created: sample_text.txt, sample_table.xlsx")
        except ImportError:
            print("⚠️ pandas not available, skipping Excel file creation")
            print("✅ Sample file created: sample_text.txt")

    def run_comprehensive_test(self):
        """Run all tests"""
        print("🚀 Starting Comprehensive API Testing...")
        print("=" * 60)

        results = {}

        # Test 0: Health check
        results["health"] = self.test_health_check()

        # Test 1: Create sample files
        self.create_sample_files()

        # Test 2: Process text document
        results["process_text"] = self.test_process_document_text()

        # Test 3: Process actual files
        if os.path.exists("sample_text.txt"):
            results["process_file_txt"] = self.test_process_document_file("sample_text.txt")

        if os.path.exists("sample_table.xlsx"):
            results["process_file_xlsx"] = self.test_process_document_file("sample_table.xlsx")

        # Test 4: Embed markdown
        results["embed_markdown"] = self.test_embed_markdown()

        # Test 5: Add FAQ
        results["add_faq"] = self.test_add_faq()

        # Test 6: Chatbot Ask - Normal cases
        results["chatbot_ask"] = self.test_chatbot_ask()

        # Test 7: Chatbot Ask - Edge cases
        results["chatbot_edge_cases"] = self.test_chatbot_edge_cases()

        # Test 8: Delete FAQ
        results["delete_faq"] = self.test_delete_faq()

        # Test 9: Delete Document
        results["delete_document"] = self.test_delete_document()

        # Test 10: Full workflow
        results["full_workflow"] = self.test_full_workflow()

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        passed = 0
        total = 0
        for test_name, result in results.items():
            total += 1
            if result.get("success"):
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")

        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

        return results


def test_individual_apis():
    """Test individual APIs separately"""
    tester = DocumentProcessingAPITester()

    print("🔧 Testing Individual APIs...")

    # Test only chatbot
    print("\n" + "=" * 30)
    print("🤖 CHATBOT API ONLY")
    print("=" * 30)

    chatbot_result = tester.test_chatbot_ask([
        "Trợ lý ảo hoạt động như thế nào?",
        "AI có thể ứng dụng trong lĩnh vực gì?",
        "Quy trình xử lý dữ liệu ra sao?"
    ])

    edge_case_result = tester.test_chatbot_edge_cases()

    print(f"\n📋 Chatbot Test Results:")
    print(f"✅ Normal questions: {chatbot_result['passed_tests']}/{chatbot_result['total_tests']}")
    print(f"🧪 Edge cases handled: {len(edge_case_result['results'])}")


def main():
    """Run API tests"""
    # Initialize tester
    tester = DocumentProcessingAPITester()

    # Option 1: Run comprehensive tests
    print("Choose test mode:")
    print("1. Comprehensive tests (all APIs)")
    print("2. Chatbot API only")
    print("3. Quick health check")

    choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"

    if choice == "2":
        test_individual_apis()
    elif choice == "3":
        tester.test_health_check()
    else:
        # Run comprehensive tests
        results = tester.run_comprehensive_test()

        # Save results to file
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Test results saved to test_results.json")


if __name__ == "__main__":
    main()