from kiri import ChunkedDocument, ElasticChunkedDocument

microservices = """
Chassis

The vision is to create a suite of microservice frameworks, or boilerplate projects, in various languages which provide a chassis for all microservices.
Features

    web server

        providing a default web server, listening on http://*:80, ready to add controllers

    structured logging

        logging at multiple levels (e.g. debug, info, warning, error, fatal) to a log sink

        support for console/stdout logging in development mode, with production-ready logging built-in

    authentication

        providing JWT authentication functionality to ensure security between all microservices

    health checks

        every service has a /health endpoint which provides at least a liveness check

        for services with external dependencies, health checks can be added for third party programs

    distributed tracing

        every request is given a correlation-id header containing a UUID

        all logs resulting from a HTTP request include the correlation-id

    metrics

        a platform for storing/tracking metrics such as RED (rate, errors, duration) and USE (utilisation, saturation, errors)

        can be gathered centrally by calling /metrics on any service

    continuous deployment

        included deployment configuration, such as a gitlab-ci.yml

    repository configuration

        standard .gitignore, .dockerignore files for consistency

        linter settings, such as .prettierrc
ng asynchronously with other services

Motivation

Such a platform would enable developers to focus on feature building whilst minimising the auxiliary operations side of the software process, hence providing higher quality features to customers more frequently.

In resulting development workflow for a microservice would be something like

    determine the purpose of the microserviceing asynchronously with other services

Motivation

Such a platform would enable developers to focus on feature building whilst minimising the auxiliary operations side of the software process, hence providing higher quality features to customers more frequently.

In resulting development workflow for a microservice would be something like

    determine the purpose of the microservice

    choose an appropriate technology from the chassis library

    fork the chassis project

    implement & test

    automatically deploy with CD pipeline
"""

branch_strategy = """
The Problem

low release cadence → many changes per release → high risk of introducing bugs → fear of releases → even lower release cadence

A self-perpetuating cycle of never wanting to release.
The Solution

In order to move fast, we must release frequently.

That’s it.

Make deployments a non-issue by deploying as often as is practical; be it every week, day, or in the ideal case, multiple times per day.
Principles
1) the master branch is always deployable

This is the only rule of the process.

Deployments should be automated with CI/CD pipelines so anything that goes into master will go straight to customers.
2) use descriptive branches off master to make changes

Use descriptive branch names to make it clear what is being worked on – the set of branches (as returned with a git fetch) should clearly describe the work in progress.

Keep branches short-lived – you don’t want to end up in ‘merge hell’.

A natural consequence, features/changes must be small; e.g. branch called implement-authentication is too broad, instead split work items down: add-signup-page, add-auth-token-header, api-auth-middleware, etc.
3) commit regularly to local branch and push your work often

Commit often to create a log of how changes were implemented.

The only branch deployments are concerned with is master, so push your work to remote regularly. This enables the branch list to describe the current work in progress.
4) get help, advice, or review by sharing a merge request

Gain insight from others by sharing work-in-progress code and discussing implementation details.
5) once merged to master, deploy it

Once work is complete, merge it and let CI/CD do the rest. If build, tests, and checks pass, it’s going live!
In Practice

master is the only consistent branch, and it must always be in a deployable state – this is the only rule.

To implement a new feature, fix a bug, or make a change,

    create a descriptive branch from master (e.g. api-key-user-id-caching)

    work on your changes locally

        commit constantly

        push to remote regularly (so everyone can see what work is in progress)

        test your changes

    create a merge request to master

        have at least one developer review your changes

        once approved, complete the merge to master

        make sure to delete the source branch and use a ‘squash’ merge to keep history tidy

    deploy immediately

        CI/CD pipelines should automatically deploy the latest changes in master – you need to be confident with your code

Notes

This strategy is heavily inspired by ‘Github Flow’

Clean up old local copies of branches that have been deleted on remote with git remote prune origin

Remove local branches that have been merged to master with git branch --merged master | grep -v '^[ *]*master$' | xargs git branch -d – it may be worth aliasing this command

    if you get the response fatal: branch name required when using this command, then there are no branches to delete
"""

standards = """
Linting

    For Python, use package autopep8 and enable 'Format on Save' in VS Code.

    For React/Javascript/Typescript, install Prettier extension for VS Code.

Code Style

    Prefer double quotes ("") to single quotes ('')

    For JSON, use snake_case for all field names

APIs

    Use _id prefix to indicate where a query parameter or field name is an identifier; e.g. GET /articles?agent_id=agnt_9p871vbtrp98bv
"""

ui_guidelines = """
Technologies

The UI is built in React using Typescript and is served as a static site, separate from the API powering it.
Dependencies

Where possible try to avoid adding extra dependencies to the project, as (in general) it reduces consistency, introduces complexity in managing package versions, and can introduce vulnerabilities or breaking changes from external sources.

All dependencies should be installed using npm install --save package-name with their corresponding type definitions installed with npm install --save-dev @types/package-name.
Configuration

    All configuration should be external to the source code, without exception.

    Environment variables and configurable constants should be provided in a .env file.

    Secrets should never be checked into source control – these must be injected at build-time using Gitlab secrets.

Component Guidelines

    All components should use the modern functional components over their class-based predecessors

    Where possible, components should be stateless for rendering data.

    Use containers for accessing and updating data from various sources, passing this down to child components for presentation.

    Every component should include a Props interface in its .tsx file indicating the expected input to any component.

    Any interface or type used by more than one component should be stored in an appropriate models definition.
"""

public_api = """
Architecture

A new public-facing service exposing article management and search functionality

Web API built in Go

Upload articles to database with POST /content, query uploaded content with POST /query

Use client key as authorization in REST call
Headers

All requests should include the X-Client-Key header with the assigned access key – this is used to identify the client hence determining which data the agent will search.

Only Content-Type: application/json should be accepted in requests, and this should be used for all responses.
"""

confluence_dev = """
Requirements

    You will need ngrok running locally (once installed, run ./ngrok http 7000 to tunnel the API port

Instructions

    Run tunnel to the API port with ./ngrok http 7000

    Replace ngrok address with the ngrok forwarding address in api/public/atlassian-connect-dev.json

    Run the API with docker-compose up --build api

    Create a PendingInstall for Confluence from frontend.

    Navigate to Confluence > Apps > Manage Apps and select Upload App

    Enter the following URL into the input field: https://YOUR_NGROK_TUNNEL_ID.ngrok.io/atlassian-connect-dev.json

    The application should install successfully

    Page creation/update events should now be received by the local API
"""

arch_design = """
The following page documents the microservice design that we should work towards as an architectural goal for the system.
Services
Dashboard

    User-facing dashboard/portal

    Management of Kiri agents

    Display usage and analytics data

    No data  stored, all served via Dashboard Gateway

Dashboard Gateway

    Internet-facing gateway service powering Dashboard web interface

    Gateway service aggregates functionality and data from many internal services to provide a single contact-point for the UI

    Responsible for checking authentication/authorization of users

    No data is persisted by this service

Public API

    Customer-facing API to power our search product

    Uses API keys as authentication, stored in a database owned by this service

User Management – Auth0

    External service providing authentication for our applications and their respective gateway services

    This service lifts the responsibility of user authentication and information security from us

Customer Service

    Internal service responsible for Customers and their details

    Includes persistent data storage for Customer models

Agent Service

    Internal service for managing agents, exposing create/delete operations

    Responsible for the Agents database

Query Service

    Internal query engine, NLP-as-a-service

    Provides an answer for a given query string and article content

    Stateless service with in-memory copy of (all) articles

    Query metrics logged to Query Analytics service

Vectorisation Service

    Internal service for vectorising articles

    No database required for this service

    Asynchronous worker processing of any articles without vectors

Query Analytics Service

    Internal service providing a centralised store of usage metrics and data

        using technology stack such as Elasticsearch

    Used by Cockpit for reporting and visualisation

    Data powers the customer-facing Dashboard for their own analytics and insights

Logz.io

    External service for log management and alerting

    All production services should log at least warnings and errors here

    Provides rolling log access and analytics through the provided Kibana interface
"""

customer_guidelines = """
Refunds
According to our refund policy, customer orders may be refunded without question if they have used their product for less than 24 hours. This can be extended to their first month if they have had a poor experience.
"""

microservices_docs = {
    "memory": ChunkedDocument(microservices, attributes={"title": "Microservices Vision"}),
    "elastic": ElasticChunkedDocument(microservices, attributes={"title": "Microservices Vision"})
}

branch_strategy_docs = {
    "memory": ChunkedDocument(branch_strategy, attributes={"title": "Git Branching Strategy"}),
    "elastic": ElasticChunkedDocument(branch_strategy, attributes={"title": "Git Branching Strategy"})
}
standards_docs = {
    "memory": ChunkedDocument(standards, attributes={"title": "Standards"}),
    "elastic": ElasticChunkedDocument(standards, attributes={"title": "Standards"})
}
ui_guidelines_docs = {
    "memory": ChunkedDocument(ui_guidelines, attributes={"title": "UI Guidelines"}),
    "elastic": ElasticChunkedDocument(ui_guidelines, attributes={"title": "UI Guidelines"})
}
public_api_docs = {
    "memory": ChunkedDocument(public_api, attributes={"title": "Public API"}),
    "elastic": ElasticChunkedDocument(public_api, attributes={"title": "Public API"})
}
confluence_dev_docs = {
    "memory": ChunkedDocument(confluence_dev, attributes={"title": "Running Confluence in Dev Mode"}),
    "elastic": ElasticChunkedDocument(confluence_dev, attributes={"title": "Running Confluence in Dev Mode"})
}
arch_design_docs = {
    "memory": ChunkedDocument(arch_design, attributes={"title": "Architecture Design"}),
    "elastic": ElasticChunkedDocument(arch_design, attributes={"title": "Architecture Design"})
}
customer_guidelines_docs = {
    "memory": ChunkedDocument(customer_guidelines, attributes={"title": "Customer Guidelines"}),
    "elastic": ElasticChunkedDocument(customer_guidelines, attributes={"title": "Customer Guidelines"})
}

tech_docs = [microservices_docs, 
             branch_strategy_docs, 
             standards_docs,
             ui_guidelines_docs,
             public_api_docs,
             confluence_dev_docs,
             arch_design_docs,
             customer_guidelines_docs]