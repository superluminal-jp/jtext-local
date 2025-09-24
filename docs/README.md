# jtext Documentation

Welcome to the jtext documentation! This directory contains comprehensive documentation for the Japanese Text Processing System.

## Documentation Structure

### 📋 [README.md](../README.md)

**Main project overview and quick start guide**

- Project overview and features
- Installation instructions
- Basic usage examples
- Architecture overview
- Quick start guide

### 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md)

**Comprehensive architecture documentation**

- Clean Architecture principles
- Layer structure and responsibilities
- Domain-Driven Design patterns
- Component interactions
- Design patterns and principles
- Testing strategy
- Future extensions

### 📚 [API_USAGE.md](API_USAGE.md)

**Python API usage guide and examples**

- Quick start examples
- Core concepts (Result types, domain objects)
- Document processing workflows
- Service configuration
- Error handling patterns
- Advanced usage scenarios
- Performance optimization
- Best practices and troubleshooting

### 👥 [USER_GUIDE.md](USER_GUIDE.md)

**End-user guide for CLI usage**

- Installation and setup
- Command reference
- Configuration options
- Output formats
- Troubleshooting guide
- Best practices
- Performance tips
- Integration examples

### 🛠️ [DEVELOPMENT.md](DEVELOPMENT.md)

**Developer guide for contributors**

- Development environment setup
- Project structure explanation
- Development workflow (TDD, Clean Architecture)
- Code standards and conventions
- Testing guidelines
- Adding new features
- Debugging techniques
- Contributing guidelines

### 🔧 [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)

**Detailed technical specification**

- System architecture details
- Domain model specification
- Infrastructure components
- Data flow diagrams
- Performance characteristics
- Security considerations
- Configuration management
- Deployment requirements

## Documentation by Audience

### 👤 **End Users**

Start with:

1. [README.md](../README.md) - Project overview
2. [USER_GUIDE.md](USER_GUIDE.md) - Complete usage guide

### 💻 **Developers Using the API**

Start with:

1. [README.md](../README.md) - Project overview
2. [API_USAGE.md](API_USAGE.md) - Programming guide

### 🧑‍💻 **Contributors**

Start with:

1. [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup
2. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
3. [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - Technical details

### 🏢 **System Architects**

Start with:

1. [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture overview
2. [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - Technical details

## Additional Resources

### 📁 Architecture Decision Records

- [ADR-001: Clean Architecture](architecture_decision_records/ADR-001-clean-architecture.md)

### 🧪 Feature Specifications

- [Document Processing Feature](../features/document_processing.feature)
- [Error Handling Feature](../features/error_handling.feature)

### 🎯 Project Status

- [FINAL_STRUCTURE_SUMMARY.md](../FINAL_STRUCTURE_SUMMARY.md) - Reorganization summary
- [COMPLETION_SUMMARY.md](../COMPLETION_SUMMARY.md) - Project completion status
- [AGENTS.md](../AGENTS.md) - Development framework and guidelines

## Quick Navigation

### Common Tasks

#### **Installing and Getting Started**

→ [README.md](../README.md#installation) → [USER_GUIDE.md](USER_GUIDE.md#quick-start)

#### **Using the Python API**

→ [API_USAGE.md](API_USAGE.md#quick-start) → [API_USAGE.md](API_USAGE.md#document-processing)

#### **Contributing to the Project**

→ [DEVELOPMENT.md](DEVELOPMENT.md#getting-started) → [DEVELOPMENT.md](DEVELOPMENT.md#development-workflow)

#### **Understanding the Architecture**

→ [ARCHITECTURE.md](ARCHITECTURE.md#overview) → [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md#system-overview)

#### **Troubleshooting Issues**

→ [USER_GUIDE.md](USER_GUIDE.md#troubleshooting) → [API_USAGE.md](API_USAGE.md#troubleshooting)

#### **Configuring the System**

→ [USER_GUIDE.md](USER_GUIDE.md#configuration) → [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md#configuration-management)

### Key Concepts

#### **Clean Architecture**

The project follows Clean Architecture principles with clear layer separation:

- **Interface Layer**: CLI, future web APIs
- **Application Layer**: Use cases, DTOs
- **Domain Layer**: Business logic, entities, value objects
- **Infrastructure Layer**: External services, data access

#### **Domain-Driven Design**

Rich domain model with:

- **Entities**: Document, ProcessingResult
- **Value Objects**: DocumentId, Confidence, FilePath
- **Domain Events**: DocumentProcessedEvent, ProcessingFailedEvent
- **Domain Services**: DocumentProcessingService

#### **Railway-Oriented Programming**

Functional error handling with Result types:

- `Ok(value)` for successful operations
- `Err(error)` for failed operations
- Chainable operations with `map()` and `and_then()`

#### **Independent Components**

Each functional area is self-contained:

- OCR processing
- Audio transcription
- LLM correction
- Event publishing
- Error handling

## Documentation Standards

### Writing Guidelines

1. **Clarity**: Use clear, simple language
2. **Examples**: Provide concrete examples for all concepts
3. **Structure**: Use consistent formatting and organization
4. **Completeness**: Cover both happy path and error scenarios
5. **Currency**: Keep documentation synchronized with code changes

### Code Examples

All code examples should:

- Be complete and runnable
- Include error handling
- Show both success and failure cases
- Use realistic data
- Follow project coding standards

### Maintenance

Documentation is maintained alongside code changes:

- Update docs when adding features
- Review docs during code reviews
- Test examples for accuracy
- Keep links and references current

## Getting Help

### Finding Information

1. **Search this documentation** for your specific use case
2. **Check the examples** in each guide
3. **Review the troubleshooting sections**
4. **Look at the test files** for additional examples

### Reporting Documentation Issues

If you find errors or omissions in the documentation:

1. Check if the issue exists in the latest version
2. Search existing issues to avoid duplicates
3. Create a new issue with:
   - Document name and section
   - Description of the problem
   - Suggested improvement

### Contributing to Documentation

We welcome documentation improvements! See [DEVELOPMENT.md](DEVELOPMENT.md#contributing) for contribution guidelines.

## License

This documentation is part of the jtext project and is licensed under the MIT License.
