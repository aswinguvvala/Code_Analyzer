# dependency_analyzer.py - Analyze project dependencies and their health

import json
import re
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from packaging import version

class DependencyAnalyzer:
    """
    Analyze project dependencies for security, freshness, and compatibility.
    
    Think of this as a health inspector for your code's "diet" - it checks
    if the external libraries you're consuming are fresh, safe, and nutritious.
    
    This analyzer:
    - Identifies all dependencies from various package files
    - Checks for known vulnerabilities
    - Analyzes update status (how outdated things are)
    - Examines license compatibility
    - Maps dependency relationships
    """
    
    def __init__(self):
        self.cache = {}  # Cache API responses to avoid hitting rate limits
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Repository-Analyzer/1.0'
        })
    
    def analyze_dependencies(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for dependency analysis.
        
        Like a comprehensive health checkup, this examines all aspects
        of your project's dependency health.
        """
        dependency_report = {
            'total_dependencies': 0,
            'dependency_files_found': [],
            'ecosystems': {},
            'vulnerabilities': [],
            'outdated_packages': [],
            'license_issues': [],
            'dependency_tree': {},
            'health_score': 0,
            'recommendations': [],
            'summary': {
                'critical_vulns': 0,
                'high_vulns': 0,
                'outdated_major': 0,
                'outdated_minor': 0,
                'license_conflicts': 0
            }
        }
        
        # Step 1: Find and parse all dependency files
        dependency_files = self._find_dependency_files(detailed_files)
        dependency_report['dependency_files_found'] = list(dependency_files.keys())
        
        # Step 2: Extract dependencies from each ecosystem
        all_dependencies = {}
        for file_path, file_info in dependency_files.items():
            ecosystem = self._detect_ecosystem(file_path)
            if ecosystem:
                deps = self._parse_dependency_file(file_path, file_info, ecosystem)
                if deps:
                    all_dependencies[ecosystem] = deps
                    dependency_report['ecosystems'][ecosystem] = {
                        'file': file_path,
                        'dependency_count': len(deps),
                        'dependencies': deps
                    }
        
        dependency_report['total_dependencies'] = sum(
            len(deps) for deps in all_dependencies.values()
        )
        
        # Step 3: Analyze each dependency for issues
        if all_dependencies:
            self._analyze_dependency_health(all_dependencies, dependency_report)
        
        # Step 4: Calculate overall health score
        dependency_report['health_score'] = self._calculate_dependency_health_score(dependency_report)
        
        # Step 5: Generate recommendations
        dependency_report['recommendations'] = self._generate_dependency_recommendations(dependency_report)
        
        return dependency_report
    
    def _find_dependency_files(self, detailed_files: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find all files that contain dependency information.
        
        Different ecosystems use different files to declare dependencies:
        - Python: requirements.txt, setup.py, pyproject.toml, Pipfile
        - JavaScript: package.json, yarn.lock, package-lock.json
        - Java: pom.xml, build.gradle
        - Ruby: Gemfile, *.gemspec
        - Go: go.mod, go.sum
        """
        dependency_files = {}
        
        dependency_file_patterns = {
            'requirements.txt': 'python',
            'setup.py': 'python',
            'pyproject.toml': 'python',
            'pipfile': 'python',
            'package.json': 'javascript',
            'yarn.lock': 'javascript',
            'package-lock.json': 'javascript',
            'pom.xml': 'java',
            'build.gradle': 'java',
            'gemfile': 'ruby',
            'go.mod': 'go',
            'composer.json': 'php',
            'cargo.toml': 'rust'
        }
        
        for file_path, file_info in detailed_files.items():
            file_name = file_path.split('/')[-1].lower()
            
            for pattern, ecosystem in dependency_file_patterns.items():
                if pattern in file_name:
                    dependency_files[file_path] = file_info
                    break
        
        return dependency_files
    
    def _detect_ecosystem(self, file_path: str) -> Optional[str]:
        """
        Determine which package ecosystem a file belongs to.
        """
        file_name = file_path.split('/')[-1].lower()
        
        if file_name in ['requirements.txt', 'setup.py', 'pyproject.toml', 'pipfile']:
            return 'python'
        elif file_name in ['package.json', 'yarn.lock', 'package-lock.json']:
            return 'javascript'
        elif file_name in ['pom.xml', 'build.gradle']:
            return 'java'
        elif file_name in ['gemfile', 'gemfile.lock']:
            return 'ruby'
        elif file_name in ['go.mod', 'go.sum']:
            return 'go'
        elif file_name in ['composer.json', 'composer.lock']:
            return 'php'
        elif file_name in ['cargo.toml', 'cargo.lock']:
            return 'rust'
        
        return None
    
    def _parse_dependency_file(self, file_path: str, file_info: Dict[str, Any], ecosystem: str) -> List[Dict[str, Any]]:
        """
        Parse dependencies from a specific file based on its ecosystem.
        
        Each ecosystem has its own format for declaring dependencies.
        This is like learning different languages - each one has its own grammar.
        """
        content = file_info.get('full_content', file_info.get('content_preview', ''))
        if not content:
            return []
        
        if ecosystem == 'python':
            return self._parse_python_dependencies(file_path, content)
        elif ecosystem == 'javascript':
            return self._parse_javascript_dependencies(file_path, content)
        elif ecosystem == 'java':
            return self._parse_java_dependencies(file_path, content)
        else:
            # For other ecosystems, we'll do basic parsing
            return self._parse_generic_dependencies(content)
    
    def _parse_python_dependencies(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Parse Python dependency files.
        
        Python has several formats:
        - requirements.txt: simple list format
        - setup.py: programmatic definition
        - pyproject.toml: modern standard
        """
        dependencies = []
        
        if 'requirements.txt' in file_path.lower():
            # Parse requirements.txt format
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    dep = self._parse_requirement_line(line)
                    if dep:
                        dependencies.append(dep)
        
        elif 'setup.py' in file_path.lower():
            # Extract install_requires from setup.py
            # This is a simplified parser - real parsing would need AST
            install_requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if install_requires_match:
                requirements_str = install_requires_match.group(1)
                for req in re.findall(r'["\']([^"\']+)["\']', requirements_str):
                    dep = self._parse_requirement_line(req)
                    if dep:
                        dependencies.append(dep)
        
        elif 'package.json' in file_path.lower():
            # This is actually JavaScript, but we handle it in the JS parser
            pass
        
        return dependencies
    
    def _parse_javascript_dependencies(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Parse JavaScript/Node.js dependency files.
        
        JavaScript primarily uses package.json with dependencies and devDependencies sections.
        """
        dependencies = []
        
        if 'package.json' in file_path.lower():
            try:
                package_data = json.loads(content)
                
                # Parse regular dependencies
                deps = package_data.get('dependencies', {})
                for name, version_spec in deps.items():
                    dependencies.append({
                        'name': name,
                        'version_spec': version_spec,
                        'ecosystem': 'npm',
                        'type': 'production',
                        'current_version': self._extract_version_from_spec(version_spec)
                    })
                
                # Parse dev dependencies
                dev_deps = package_data.get('devDependencies', {})
                for name, version_spec in dev_deps.items():
                    dependencies.append({
                        'name': name,
                        'version_spec': version_spec,
                        'ecosystem': 'npm',
                        'type': 'development',
                        'current_version': self._extract_version_from_spec(version_spec)
                    })
            
            except json.JSONDecodeError:
                # If JSON parsing fails, we can't extract dependencies
                pass
        
        return dependencies
    
    def _parse_java_dependencies(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Parse Java dependency files (Maven POM or Gradle).
        
        This is more complex because Java build files can have complex structures.
        We'll do basic parsing for common patterns.
        """
        dependencies = []
        
        if 'pom.xml' in file_path.lower():
            # Parse Maven POM dependencies
            # Look for <dependency> blocks
            dependency_pattern = r'<dependency>\s*<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>\s*<version>([^<]+)</version>'
            matches = re.findall(dependency_pattern, content, re.IGNORECASE)
            
            for group_id, artifact_id, version_spec in matches:
                dependencies.append({
                    'name': f"{group_id}:{artifact_id}",
                    'version_spec': version_spec,
                    'ecosystem': 'maven',
                    'type': 'production',
                    'current_version': version_spec.strip()
                })
        
        elif 'build.gradle' in file_path.lower():
            # Parse Gradle dependencies
            # Look for implementation, compile, etc. declarations
            gradle_dep_pattern = r'(?:implementation|compile|api|testImplementation)\s+["\']([^:"\']+):([^:"\']+):([^"\']+)["\']'
            matches = re.findall(gradle_dep_pattern, content)
            
            for group_id, artifact_id, version_spec in matches:
                dependencies.append({
                    'name': f"{group_id}:{artifact_id}",
                    'version_spec': version_spec,
                    'ecosystem': 'maven',
                    'type': 'production',
                    'current_version': version_spec.strip()
                })
        
        return dependencies
    
    def _parse_generic_dependencies(self, content: str) -> List[Dict[str, Any]]:
        """
        Generic dependency parser for ecosystems we don't have specific support for.
        """
        # This is a placeholder for other ecosystems
        # In a real implementation, you'd add parsers for Go, Ruby, etc.
        return []
    
    def _parse_requirement_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single requirement line from requirements.txt format.
        
        Examples:
        - django==3.2.0
        - requests>=2.25.0
        - flask~=2.0.0
        - pytest  # latest version
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        # Remove comments
        if '#' in line:
            line = line.split('#')[0].strip()
        
        # Parse name and version specification
        version_operators = ['==', '>=', '<=', '>', '<', '~=', '!=']
        
        for op in version_operators:
            if op in line:
                parts = line.split(op)
                if len(parts) == 2:
                    name = parts[0].strip()
                    version_spec = op + parts[1].strip()
                    current_version = parts[1].strip()
                    
                    return {
                        'name': name,
                        'version_spec': version_spec,
                        'ecosystem': 'pip',
                        'type': 'production',
                        'current_version': current_version
                    }
        
        # No version specified, assume latest
        return {
            'name': line.strip(),
            'version_spec': '*',
            'ecosystem': 'pip',
            'type': 'production',
            'current_version': None
        }
    
    def _extract_version_from_spec(self, version_spec: str) -> Optional[str]:
        """
        Extract actual version number from a version specification.
        
        Examples:
        - "^3.2.0" -> "3.2.0"
        - "~2.1.5" -> "2.1.5"
        - ">=1.0.0" -> "1.0.0"
        """
        # Remove common prefixes
        cleaned = re.sub(r'^[\^~>=<!=]+', '', version_spec.strip())
        
        # Extract version number
        version_pattern = r'(\d+(?:\.\d+)*(?:\.\d+)*)'
        match = re.search(version_pattern, cleaned)
        
        return match.group(1) if match else None
    
    def _analyze_dependency_health(self, all_dependencies: Dict[str, List[Dict]], dependency_report: Dict[str, Any]):
        """
        Analyze the health of all dependencies.
        
        This is like getting a full health panel - we check multiple indicators
        to understand the overall health of your dependency ecosystem.
        """
        for ecosystem, dependencies in all_dependencies.items():
            for dep in dependencies:
                # Check for known vulnerabilities
                vulnerabilities = self._check_vulnerabilities(dep)
                dependency_report['vulnerabilities'].extend(vulnerabilities)
                
                # Check if package is outdated
                outdated_info = self._check_if_outdated(dep)
                if outdated_info:
                    dependency_report['outdated_packages'].append(outdated_info)
                
                # Check license compatibility
                license_info = self._check_license(dep)
                if license_info and license_info.get('has_issues'):
                    dependency_report['license_issues'].append(license_info)
        
        # Update summary counts
        dependency_report['summary']['critical_vulns'] = len([v for v in dependency_report['vulnerabilities'] if v.get('severity') == 'critical'])
        dependency_report['summary']['high_vulns'] = len([v for v in dependency_report['vulnerabilities'] if v.get('severity') == 'high'])
        dependency_report['summary']['outdated_major'] = len([p for p in dependency_report['outdated_packages'] if p.get('update_type') == 'major'])
        dependency_report['summary']['outdated_minor'] = len([p for p in dependency_report['outdated_packages'] if p.get('update_type') == 'minor'])
        dependency_report['summary']['license_conflicts'] = len(dependency_report['license_issues'])
    
    def _check_vulnerabilities(self, dependency: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if a dependency has known security vulnerabilities.
        
        This is like checking if any of your food has been recalled for safety issues.
        We use public vulnerability databases to check for known security problems.
        """
        vulnerabilities = []
        
        # For Python packages, we could use the PyPI JSON API and cross-reference with vulnerability databases
        # For this example, we'll implement a basic check
        
        if dependency['ecosystem'] in ['pip', 'npm']:
            # Simulate vulnerability checking
            # In a real implementation, you'd query databases like:
            # - GitHub Advisory Database
            # - National Vulnerability Database (NVD)
            # - Snyk vulnerability database
            # - PyUp.io safety database
            
            known_vulnerable_packages = {
                'django': ['1.11.0', '2.0.0', '2.1.0'],
                'requests': ['2.19.0', '2.20.0'],
                'lodash': ['4.17.19', '4.17.20'],
                'express': ['4.16.0', '4.17.0']
            }
            
            package_name = dependency['name'].lower()
            current_version = dependency.get('current_version')
            
            if package_name in known_vulnerable_packages and current_version:
                vulnerable_versions = known_vulnerable_packages[package_name]
                
                for vuln_version in vulnerable_versions:
                    try:
                        if version.parse(current_version) <= version.parse(vuln_version):
                            vulnerabilities.append({
                                'package': dependency['name'],
                                'current_version': current_version,
                                'vulnerable_version': vuln_version,
                                'severity': 'high',  # Would come from vulnerability database
                                'description': f"Known vulnerability in {package_name} version {vuln_version}",
                                'recommendation': f"Update {package_name} to a version newer than {vuln_version}"
                            })
                    except:
                        # Skip if version parsing fails
                        pass
        
        return vulnerabilities
    
    def _check_if_outdated(self, dependency: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if a dependency is outdated compared to the latest available version.
        
        This is like checking if your medications are expired or if there are
        newer, better versions available.
        """
        current_version = dependency.get('current_version')
        if not current_version:
            return None
        
        # Get latest version from package registry
        latest_version = self._get_latest_version(dependency)
        if not latest_version:
            return None
        
        try:
            current_ver = version.parse(current_version)
            latest_ver = version.parse(latest_version)
            
            if latest_ver > current_ver:
                # Determine update type
                update_type = 'patch'
                if latest_ver.major > current_ver.major:
                    update_type = 'major'
                elif latest_ver.minor > current_ver.minor:
                    update_type = 'minor'
                
                return {
                    'package': dependency['name'],
                    'current_version': current_version,
                    'latest_version': latest_version,
                    'update_type': update_type,
                    'ecosystem': dependency['ecosystem'],
                    'recommendation': self._get_update_recommendation(update_type, dependency['name'])
                }
        
        except Exception:
            # Skip if version comparison fails
            pass
        
        return None
    
    def _get_latest_version(self, dependency: Dict[str, Any]) -> Optional[str]:
        """
        Get the latest version of a package from its registry.
        
        We cache results to avoid hitting API rate limits.
        """
        package_name = dependency['name']
        ecosystem = dependency['ecosystem']
        
        # Check cache first
        cache_key = f"{ecosystem}:{package_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        latest_version = None
        
        try:
            if ecosystem == 'pip':
                # Query PyPI JSON API
                response = self.session.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    latest_version = data['info']['version']
            
            elif ecosystem == 'npm':
                # Query npm registry
                response = self.session.get(f"https://registry.npmjs.org/{package_name}/latest", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    latest_version = data['version']
            
            # Cache the result
            if latest_version:
                self.cache[cache_key] = latest_version
                
        except Exception:
            # If we can't fetch the latest version, that's okay
            pass
        
        return latest_version
    
    def _check_license(self, dependency: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check the license of a dependency for compatibility issues.
        
        This is like checking if you're allowed to use certain ingredients
        in your recipe based on dietary restrictions or regulations.
        """
        # This is a simplified implementation
        # In practice, you'd need to query package metadata and analyze license compatibility
        
        problematic_licenses = [
            'GPL-3.0',    # Copyleft license that might conflict with proprietary code
            'AGPL-3.0',   # Strong copyleft license
            'GPL-2.0'     # Another copyleft license
        ]
        
        # For this example, we'll return basic license info
        # In a real implementation, you'd fetch actual license information from package registries
        
        return None  # Placeholder - would return license analysis in real implementation
    
    def _get_update_recommendation(self, update_type: str, package_name: str) -> str:
        """
        Generate appropriate update recommendations based on the type of update.
        """
        if update_type == 'major':
            return f"Major version update available for {package_name}. Review changelog for breaking changes before updating."
        elif update_type == 'minor':
            return f"Minor version update available for {package_name}. Should be safe to update, but test thoroughly."
        else:
            return f"Patch update available for {package_name}. Generally safe to update immediately."
    
    def _calculate_dependency_health_score(self, dependency_report: Dict[str, Any]) -> int:
        """
        Calculate an overall health score for the dependency ecosystem.
        
        Like a health score that considers multiple factors:
        - How many vulnerabilities exist
        - How outdated packages are
        - License compatibility issues
        """
        if dependency_report['total_dependencies'] == 0:
            return 100  # No dependencies = no problems
        
        score = 100
        summary = dependency_report['summary']
        
        # Deduct points for vulnerabilities
        score -= summary['critical_vulns'] * 25
        score -= summary['high_vulns'] * 15
        
        # Deduct points for outdated packages
        score -= summary['outdated_major'] * 10
        score -= summary['outdated_minor'] * 5
        
        # Deduct points for license issues
        score -= summary['license_conflicts'] * 5
        
        return max(0, score)
    
    def _generate_dependency_recommendations(self, dependency_report: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations for improving dependency health.
        """
        recommendations = []
        summary = dependency_report['summary']
        
        if summary['critical_vulns'] > 0:
            recommendations.append(f"🚨 URGENT: Fix {summary['critical_vulns']} critical vulnerabilities immediately")
        
        if summary['high_vulns'] > 0:
            recommendations.append(f"⚠️ Address {summary['high_vulns']} high-severity vulnerabilities")
        
        if summary['outdated_major'] > 0:
            recommendations.append(f"📈 Consider updating {summary['outdated_major']} packages with major version updates (review breaking changes)")
        
        if summary['outdated_minor'] > 0:
            recommendations.append(f"🔄 Update {summary['outdated_minor']} packages with minor version updates")
        
        if dependency_report['total_dependencies'] > 50:
            recommendations.append("📦 Consider dependency cleanup - large dependency trees increase attack surface")
        
        # Add general recommendations
        if dependency_report['vulnerabilities'] or dependency_report['outdated_packages']:
            recommendations.append("🔧 Set up automated dependency scanning in your CI/CD pipeline")
            recommendations.append("📅 Schedule regular dependency updates")
        
        if not recommendations:
            recommendations.append("✅ Great job! Your dependencies look healthy")
        
        return recommendations