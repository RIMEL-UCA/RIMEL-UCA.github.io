package fr.unice.polytech.si5.rimel.eventprofile.repository;

import fr.unice.polytech.si5.rimel.eventprofile.domain.Orm;
import fr.unice.polytech.si5.rimel.eventprofile.domain.StackType;
import fr.unice.polytech.si5.rimel.eventprofile.model.CI;
import fr.unice.polytech.si5.rimel.eventprofile.model.Docker;
import fr.unice.polytech.si5.rimel.eventprofile.model.HasTechnology;

import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface IHasTechnology extends Neo4jRepository<HasTechnology, String> {

    Optional<HasTechnology> findByTechnology_Stack_AndNameAndPom_FullRepositoryName(StackType stackType, String name, String pomName);
    Optional<HasTechnology> findByTechnology_StackAndPom_FullRepositoryName(StackType stackType, String pomName);

}