# Microservices Architecture
Oposed to a monolithic architecture, it enables and facilitates CI, CD, project management and reliability.

## Best practices
- Secure or limit access and communications
- Stateless services are desired
- Develop to be tested
- Implement to not rely on base platforms or hosts: bare metal, cloud, etc.
    - Avoid depending on host configurations
- What happens if a project requirement changes?
    - A change in a requirement should only impact a unique service
    - Services should be organised based on responsabilities and it should change only if its responsability changes 
- What happens if a interface changes?
    - Services should be loosely coupled
    - Services should ensure backward compatibility (previous versions)
    - Services waiting info/commands (SUB / server):
        - Services should not fail on unexpected or extra sets of data
        - Services should not rely on complete messages, sub-sets of them should be valid to operate
    - Services sending info/commands (PUB / server):
        - Do not overload messages, they whould be simple, sufficient and as small as possible
        - Explose comprehensive data sets
        - Binary data might be difficult to process by certain languages
        - Float values might loose accuracy when stored or passed to different services
        - Integer values are easier to filter and process
        - Clarify value ranges
- How can be increased a service capability?
    - If a service can be replicated, ensure it is stateless and that no global variables must be also replicated in every replica
- Need data or status be persistent?
    - Each service manage their data: creation of data store structure, maintain and update their data
- What happen if an interface does not exists or breaks down?
    - Avoid blocking the code execution
    - Enable other dependent microservices to continue its execution by retrieving or sending default and conservative data
- What happens if a service fails?
    - Avoid single points of failure 
- What happens when another version of a service is deployed?
    - Persistent data store structures should be uploaded by their services
    - Downgrades should not be affected by uploaded interfaces
- What happens if a configuration file version changes?
    - Set internal default values to avoid updating host files on production
    - Choose internal default values that do not affect to previous distributions

## Communication protocols
- MQTT: pub-sub protocol. Easy to define, lightweight and event based.
    - REF: https://github.com/mqtt/mqtt.github.io/wiki/software?id=software
    - It is a bottle neck in robustness of a micreservice architecture since is a unique point of failure
- gRPC: server-client protocol. Efficient and fast.

## Persistant Data
- REF: https://www.sciencedirect.com/topics/computer-science/persistence-service
- DB based: 
(research)

## Platforms / Hosts

- Ansible 
(research)
