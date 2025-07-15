# security_scanner.py - Security vulnerability detection module

import re
import json
import hashlib
from typing import Dict, List, Any, Tuple, Set
from pathlib import Path

class SecurityScanner:
    """
    This class scans code for common security vulnerabilities.
    
    Think of this as a security guard that knows what to look for:
    - Hardcoded secrets (like leaving your house key under the doormat)
    - SQL injection risks (like leaving doors unlocked)
    - Unsafe file operations (like letting strangers into your house)
    
    The scanner uses pattern matching - like teaching a guard to recognize
    suspicious behavior by showing them examples of what to watch for.
    """
    
    def __init__(self):
        # These patterns are like a "wanted poster" for security vulnerabilities
        self.secret_patterns = self._build_secret_patterns()
        self.vulnerability_patterns = self._build_vulnerability_patterns()
        self.safe_contexts = self._build_safe_contexts()
    
    def _build_secret_patterns(self) -> Dict[str, re.Pattern]:
        """
        Build regex patterns to detect hardcoded secrets.
        
        Hardcoded secrets are like writing your password on a sticky note -
        they're visible to anyone who can see your code.
        """
        patterns = {
            'api_key': re.compile(r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']([A-Za-z0-9\-_]{16,})["\']'),
            'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
            'aws_secret_key': re.compile(r'(?i)(aws[_-]?secret|secret[_-]?key)\s*[=:]\s*["\']([A-Za-z0-9/+=]{40})["\']'),
            'github_token': re.compile(r'ghp_[A-Za-z0-9]{36}'),
            'slack_token': re.compile(r'xox[baprs]-[A-Za-z0-9\-]{10,48}'),
            'private_key': re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'),
            'password': re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\'\s]{8,})["\']'),
            'database_url': re.compile(r'(?i)(database[_-]?url|db[_-]?url)\s*[=:]\s*["\']([^"\'\s]+://[^"\'\s]+)["\']'),
            'jwt_secret': re.compile(r'(?i)(jwt[_-]?secret|secret[_-]?key)\s*[=:]\s*["\']([A-Za-z0-9\-_=]{32,})["\']'),
            'encryption_key': re.compile(r'(?i)(encryption[_-]?key|secret[_-]?key)\s*[=:]\s*["\']([A-Fa-f0-9]{32,})["\']')
        }
        return patterns
    
    def _build_vulnerability_patterns(self) -> Dict[str, Dict[str, re.Pattern]]:
        """
        Build patterns for detecting common vulnerability types.
        
        These are like security checkpoints for different types of risky code.
        Each vulnerability type has its own "signature" that we can recognize.
        """
        patterns = {
            'sql_injection': {
                'python': re.compile(r'(?i)(execute|query|cursor\.execute)\s*\(\s*["\'].*%s.*["\'].*%'),
                'javascript': re.compile(r'(?i)(query|execute)\s*\(\s*["`\'].*\+.*["`\']'),
                'general': re.compile(r'(?i)(select|insert|update|delete).*\+.*["\']')
            },
            'xss': {
                'javascript': re.compile(r'(?i)(innerHTML|outerHTML)\s*=\s*.*\+'),
                'python': re.compile(r'(?i)(render_template_string|safe)\s*\(.*\+'),
                'general': re.compile(r'(?i)(eval|exec)\s*\(.*user.*\)')
            },
            'path_traversal': {
                'python': re.compile(r'(?i)(open|file)\s*\(.*\+.*request'),
                'javascript': re.compile(r'(?i)(readFile|writeFile)\s*\(.*req\.'),
                'general': re.compile(r'(?i)\.\./.*\.\.')
            },
            'command_injection': {
                'python': re.compile(r'(?i)(os\.system|subprocess|exec|eval)\s*\(.*\+'),
                'javascript': re.compile(r'(?i)(exec|spawn|child_process)\s*\(.*\+'),
                'general': re.compile(r'(?i)(system|exec|shell)\s*\(.*user')
            },
            'insecure_random': {
                'python': re.compile(r'(?i)random\.(random|randint|choice)'),
                'javascript': re.compile(r'(?i)Math\.random\(\)'),
                'general': re.compile(r'(?i)rand\(\)')
            },
            'hardcoded_crypto': {
                'python': re.compile(r'(?i)(key|iv|salt)\s*=\s*["\'][A-Fa-f0-9]{16,}["\']'),
                'javascript': re.compile(r'(?i)(key|iv|salt)\s*=\s*["`\'][A-Fa-f0-9]{16,}["`\']'),
                'general': re.compile(r'(?i)(aes|des|md5|sha1).*["\'][A-Fa-f0-9]+["\']')
            }
        }
        return patterns
    
    def _build_safe_contexts(self) -> Set[str]:
        """
        Define contexts where potential vulnerabilities might be false positives.
        
        This is like teaching our security guard to not worry about certain situations -
        for example, seeing "password" in a comment about password validation is different
        from seeing an actual hardcoded password.
        """
        return {
            'comment',      # Things in comments are usually examples or explanations
            'test',         # Test files often have dummy data
            'example',      # Example code typically uses placeholder values
            'documentation',# Documentation often shows examples
            'template',     # Templates may have placeholder patterns
            'config',       # Config files might have example values
        }
    
    def scan_repository(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan the entire repository for security vulnerabilities.
        
        This is our main security checkpoint - every file goes through
        multiple security checks, like airport security but for code.
        """
        security_report = {
            'total_files_scanned': 0,
            'vulnerabilities_found': 0,
            'secrets_found': 0,
            'risk_level': 'low',
            'findings': {
                'secrets': [],
                'vulnerabilities': [],
                'recommendations': []
            },
            'risk_summary': {
                'critical': 0,
                'high': 0, 
                'medium': 0,
                'low': 0
            }
        }
        
        for file_path, file_info in detailed_files.items():
            # Skip non-code files and very large files
            if not self._should_scan_file(file_path, file_info):
                continue
            
            security_report['total_files_scanned'] += 1
            content = file_info.get('full_content', file_info.get('content_preview', ''))
            
            if not content:
                continue
            
            # Scan for hardcoded secrets
            secrets = self._scan_for_secrets(file_path, content)
            security_report['findings']['secrets'].extend(secrets)
            security_report['secrets_found'] += len(secrets)
            
            # Scan for vulnerabilities
            vulns = self._scan_for_vulnerabilities(file_path, content, file_info.get('extension', ''))
            security_report['findings']['vulnerabilities'].extend(vulns)
            security_report['vulnerabilities_found'] += len(vulns)
        
        # Calculate overall risk level
        security_report['risk_level'] = self._calculate_risk_level(security_report)
        
        # Generate recommendations
        security_report['findings']['recommendations'] = self._generate_security_recommendations(security_report)
        
        # Update risk summary
        for finding in security_report['findings']['secrets'] + security_report['findings']['vulnerabilities']:
            risk = finding.get('risk_level', 'low')
            security_report['risk_summary'][risk] += 1
        
        return security_report
    
    def _should_scan_file(self, file_path: str, file_info: Dict[str, Any]) -> bool:
        """
        Determine if a file should be scanned for security issues.
        
        We focus on files that are likely to contain security-relevant code.
        Like a security guard focusing on areas where valuable things are stored.
        """
        # Skip binary files, very large files, and irrelevant file types
        extension = file_info.get('extension', '').lower()
        
        # Scan code files and configuration files
        scannable_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.php', '.rb', '.go', '.rs',
            '.json', '.yaml', '.yml', '.env', '.config', '.cfg', '.ini', '.sql',
            '.sh', '.bash', '.ps1', '.xml', '.properties'
        }
        
        if extension not in scannable_extensions:
            return False
        
        # Skip very large files (over 10,000 lines) to avoid performance issues
        if file_info.get('lines', 0) > 10000:
            return False
        
        # Skip files that are clearly test fixtures or examples
        if any(word in file_path.lower() for word in ['test_fixtures', 'example_data', 'sample_', 'mock_']):
            return False
        
        return True
    
    def _scan_for_secrets(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Scan file content for hardcoded secrets.
        
        This is like checking every pocket and bag for items that shouldn't be there.
        We look for patterns that match common secret formats.
        """
        secrets = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip if line appears to be in a safe context
            if self._is_safe_context(line):
                continue
            
            for secret_type, pattern in self.secret_patterns.items():
                matches = pattern.findall(line)
                if matches:
                    # For patterns that capture groups, extract the actual secret
                    if isinstance(matches[0], tuple):
                        secret_value = matches[0][-1]  # Last group is usually the secret
                    else:
                        secret_value = matches[0]
                    
                    # Calculate entropy to reduce false positives
                    # High entropy suggests it's likely a real secret, not an example
                    entropy = self._calculate_entropy(secret_value)
                    
                    # Determine risk level based on secret type and entropy
                    risk_level = self._assess_secret_risk(secret_type, secret_value, entropy, file_path)
                    
                    secrets.append({
                        'type': 'secret',
                        'subtype': secret_type,
                        'file': file_path,
                        'line': line_num,
                        'content': line.strip(),
                        'secret_value': secret_value[:10] + '...' if len(secret_value) > 10 else secret_value,
                        'entropy': entropy,
                        'risk_level': risk_level,
                        'description': f"Potential {secret_type.replace('_', ' ')} found",
                        'recommendation': self._get_secret_recommendation(secret_type)
                    })
        
        return secrets
    
    def _scan_for_vulnerabilities(self, file_path: str, content: str, file_extension: str) -> List[Dict[str, Any]]:
        """
        Scan for common code vulnerabilities.
        
        This looks for coding patterns that create security risks.
        Like checking for unlocked doors or windows in a building.
        """
        vulnerabilities = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and safe contexts
            if self._is_safe_context(line):
                continue
            
            for vuln_type, patterns in self.vulnerability_patterns.items():
                # Check language-specific patterns first, then general patterns
                pattern_to_use = None
                
                if file_extension == '.py' and 'python' in patterns:
                    pattern_to_use = patterns['python']
                elif file_extension in ['.js', '.ts', '.jsx', '.tsx'] and 'javascript' in patterns:
                    pattern_to_use = patterns['javascript']
                elif 'general' in patterns:
                    pattern_to_use = patterns['general']
                
                if pattern_to_use and pattern_to_use.search(line):
                    risk_level = self._assess_vulnerability_risk(vuln_type, line, file_path)
                    
                    vulnerabilities.append({
                        'type': 'vulnerability',
                        'subtype': vuln_type,
                        'file': file_path,
                        'line': line_num,
                        'content': line.strip(),
                        'risk_level': risk_level,
                        'description': self._get_vulnerability_description(vuln_type),
                        'recommendation': self._get_vulnerability_recommendation(vuln_type)
                    })
        
        return vulnerabilities
    
    def _is_safe_context(self, line: str) -> bool:
        """
        Check if a line is in a context where findings would be false positives.
        
        This prevents our scanner from crying wolf about harmless code.
        """
        line_lower = line.lower().strip()
        
        # Skip comments
        if line_lower.startswith('#') or line_lower.startswith('//') or line_lower.startswith('/*'):
            return True
        
        # Skip lines that look like examples or documentation
        if any(word in line_lower for word in ['example', 'sample', 'test', 'mock', 'dummy', 'placeholder']):
            return True
        
        # Skip variable declarations that are clearly examples
        if 'example' in line_lower or 'sample' in line_lower:
            return True
        
        return False
    
    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text to assess randomness.
        
        Real secrets tend to have high entropy (randomness).
        Examples like "your_api_key_here" have low entropy.
        """
        if not text:
            return 0
        
        # Count frequency of each character
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        text_len = len(text)
        for count in char_counts.values():
            probability = count / text_len
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    def _assess_secret_risk(self, secret_type: str, secret_value: str, entropy: float, file_path: str) -> str:
        """
        Assess the risk level of a potential secret.
        
        Like a security guard deciding how urgent a situation is.
        """
        # High-risk secret types
        if secret_type in ['private_key', 'aws_secret_key', 'github_token']:
            return 'critical'
        
        # Check if it looks like a real secret vs an example
        if entropy < 2.0:  # Low entropy suggests it's probably an example
            return 'low'
        
        # Check file context
        if any(word in file_path.lower() for word in ['test', 'example', 'sample', 'demo']):
            return 'low'
        
        # Check if secret value looks real
        if any(word in secret_value.lower() for word in ['example', 'sample', 'test', 'dummy', 'placeholder']):
            return 'low'
        
        # Medium-high risk for other real-looking secrets
        if secret_type in ['api_key', 'password', 'database_url']:
            return 'high' if entropy > 3.0 else 'medium'
        
        return 'medium'
    
    def _assess_vulnerability_risk(self, vuln_type: str, line: str, file_path: str) -> str:
        """
        Assess the risk level of a potential vulnerability.
        """
        # SQL injection and command injection are typically high risk
        if vuln_type in ['sql_injection', 'command_injection']:
            return 'high'
        
        # XSS and path traversal are medium-high risk
        if vuln_type in ['xss', 'path_traversal']:
            return 'medium'
        
        # Insecure random and hardcoded crypto are medium risk
        if vuln_type in ['insecure_random', 'hardcoded_crypto']:
            return 'medium'
        
        return 'low'
    
    def _calculate_risk_level(self, security_report: Dict[str, Any]) -> str:
        """
        Calculate overall repository risk level.
        
        Like giving an overall security grade to the entire codebase.
        """
        findings = security_report['findings']
        total_critical = sum(1 for f in findings['secrets'] + findings['vulnerabilities'] if f.get('risk_level') == 'critical')
        total_high = sum(1 for f in findings['secrets'] + findings['vulnerabilities'] if f.get('risk_level') == 'high')
        total_medium = sum(1 for f in findings['secrets'] + findings['vulnerabilities'] if f.get('risk_level') == 'medium')
        
        if total_critical > 0:
            return 'critical'
        elif total_high > 2:
            return 'high'
        elif total_high > 0 or total_medium > 3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_security_recommendations(self, security_report: Dict[str, Any]) -> List[str]:
        """
        Generate actionable security recommendations.
        
        Like a security consultant giving you a prioritized action plan.
        """
        recommendations = []
        findings = security_report['findings']
        
        # Count different types of issues
        secret_count = len(findings['secrets'])
        vuln_count = len(findings['vulnerabilities'])
        
        if secret_count > 0:
            recommendations.append(f"🚨 Found {secret_count} potential secrets - move these to environment variables or secure vaults")
            recommendations.append("🔐 Use tools like git-secrets or pre-commit hooks to prevent future secret commits")
        
        if vuln_count > 0:
            recommendations.append(f"⚠️ Found {vuln_count} potential vulnerabilities - review and fix high-risk issues first")
        
        # Specific recommendations based on vulnerability types found
        vuln_types = set(f['subtype'] for f in findings['vulnerabilities'])
        
        if 'sql_injection' in vuln_types:
            recommendations.append("🛡️ Use parameterized queries or ORMs to prevent SQL injection")
        
        if 'xss' in vuln_types:
            recommendations.append("🔒 Sanitize user input and use proper escaping for output")
        
        if 'command_injection' in vuln_types:
            recommendations.append("⚡ Avoid dynamic command execution with user input")
        
        if 'insecure_random' in vuln_types:
            recommendations.append("🎲 Use cryptographically secure random number generators for security-sensitive operations")
        
        # General recommendations
        if secret_count > 0 or vuln_count > 0:
            recommendations.append("📚 Consider integrating security scanning into your CI/CD pipeline")
            recommendations.append("👥 Conduct security code reviews for sensitive components")
        
        if not recommendations:
            recommendations.append("✅ Great job! No obvious security issues found in the scanned files")
        
        return recommendations
    
    def _get_secret_recommendation(self, secret_type: str) -> str:
        """Get specific recommendation for a type of secret."""
        recommendations = {
            'api_key': "Move to environment variables or secure key management service",
            'aws_access_key': "Use IAM roles or AWS credentials file instead of hardcoding",
            'github_token': "Use GitHub secrets or environment variables",
            'password': "Use environment variables or secure configuration management",
            'database_url': "Store in environment variables with proper access controls",
            'private_key': "Store in secure key management system, never in code",
            'jwt_secret': "Use environment variables and rotate regularly"
        }
        return recommendations.get(secret_type, "Move to secure configuration management")
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get description of what the vulnerability means."""
        descriptions = {
            'sql_injection': "Potential SQL injection vulnerability - user input may be directly concatenated into SQL queries",
            'xss': "Potential cross-site scripting (XSS) vulnerability - user input may be rendered without proper escaping",
            'path_traversal': "Potential path traversal vulnerability - file paths may be constructed from user input",
            'command_injection': "Potential command injection vulnerability - user input may be passed to system commands",
            'insecure_random': "Use of insecure random number generation for security-sensitive operations",
            'hardcoded_crypto': "Hardcoded cryptographic keys or initialization vectors found"
        }
        return descriptions.get(vuln_type, "Potential security vulnerability detected")
    
    def _get_vulnerability_recommendation(self, vuln_type: str) -> str:
        """Get specific recommendation for a type of vulnerability."""
        recommendations = {
            'sql_injection': "Use parameterized queries or prepared statements",
            'xss': "Sanitize input and escape output properly",
            'path_traversal': "Validate and sanitize file paths, use allow-lists",
            'command_injection': "Avoid dynamic command construction, use safe APIs",
            'insecure_random': "Use cryptographically secure random number generators",
            'hardcoded_crypto': "Use secure key management and generate keys dynamically"
        }
        return recommendations.get(vuln_type, "Review and address this security concern")